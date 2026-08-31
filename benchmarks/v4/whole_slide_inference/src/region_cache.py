"""Lossless-on-dtype storage for prompt-independent J5 fine regions."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from benchmarks.v4.whole_slide_inference.src.prompt_transfer import EncodedRegions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RegionCacheStore:
    """Read exact Phase2/3 outputs and rebuild J5's full assignment verbatim."""

    def __init__(self, manifest_path: str | Path, expected_rows: int, max_open_shards: int = 64):
        self.manifest_path = Path(manifest_path)
        self.manifest = pd.read_parquet(self.manifest_path).sort_values("start").reset_index(drop=True)
        required = {"shard_path", "start", "end", "rows", "sha256"}
        if not required.issubset(self.manifest.columns):
            raise ValueError(f"region cache manifest misses {sorted(required - set(self.manifest.columns))}")
        starts = self.manifest["start"].to_numpy(dtype="int64")
        self.ends = self.manifest["end"].to_numpy(dtype="int64")
        if len(self.manifest) == 0 or starts[0] != 0 or self.ends[-1] != int(expected_rows):
            raise ValueError("region cache does not cover the complete tile index")
        if len(self.manifest) > 1 and not (starts[1:] == self.ends[:-1]).all():
            raise ValueError("region cache has a gap or overlap")
        self.paths = [Path(value) for value in self.manifest["shard_path"]]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"region cache shard missing: {missing[0]}")
        self.cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")

    def preload(self) -> int:
        """Load every shard when the configured RAM cache can retain the bundle."""
        if len(self.paths) > self.max_open_shards:
            return 0
        for shard_index in range(len(self.paths)):
            self._payload(shard_index)
        return len(self.paths)

    def _payload(self, shard_index: int) -> dict[str, torch.Tensor]:
        path = self.paths[shard_index]
        payload = self.cache.get(path)
        if payload is None:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.cache[path] = payload
            if len(self.cache) > self.max_open_shards:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(path)
        return payload

    def encoded(self, rows: list[dict], device: torch.device) -> EncodedRegions:
        values: dict[str, list[torch.Tensor]] = {key: [] for key in ("assignment_low", "tokens", "xy", "area", "active")}
        for row in rows:
            source_index = int(row["source_index"])
            shard_index = int(self.ends.searchsorted(source_index, side="right"))
            payload = self._payload(shard_index)
            offset = source_index - int(self.manifest.iloc[shard_index].start)
            if int(payload["source_index"][offset]) != source_index:
                raise RuntimeError(f"region cache identity mismatch at {source_index}")
            values["assignment_low"].append(payload["assignment_low"][offset])
            values["tokens"].append(payload["tokens"][offset])
            values["xy"].append(payload["xy"][offset])
            values["area"].append(payload["area"][offset])
            values["active"].append(payload["active"][offset])
        low = torch.stack(values["assignment_low"]).to(device, non_blocking=True)
        assignment = F.interpolate(low, size=(int(rows[0]["height_10x"]), int(rows[0]["width_10x"])), mode="bilinear", align_corners=False)
        assignment = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-6)
        return EncodedRegions(
            assignment=assignment,
            tokens=torch.stack(values["tokens"]).to(device, non_blocking=True),
            xy=torch.stack(values["xy"]).to(device, non_blocking=True),
            area=torch.stack(values["area"]).to(device, non_blocking=True),
            active=torch.stack(values["active"]).to(device, non_blocking=True),
            assignment_low=low,
        )

    def prompt_encoded(self, row: dict, device: torch.device) -> EncodedRegions:
        """Return the formal batch=1 representation used for prompt selection."""
        source_index = int(row["source_index"])
        shard_index = int(self.ends.searchsorted(source_index, side="right"))
        payload = self._payload(shard_index)
        offset = source_index - int(self.manifest.iloc[shard_index].start)
        if int(payload["source_index"][offset]) != source_index:
            raise RuntimeError(f"region cache identity mismatch at {source_index}")
        required = ("prompt_tokens", "prompt_xy", "prompt_area", "prompt_active")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(
                "region cache lacks formal batch=1 prompt representations; rebuild the cache"
            )
        return EncodedRegions(
            assignment=torch.empty((1, 0, 0, 0), device=device),
            tokens=payload["prompt_tokens"][offset:offset + 1].to(device, non_blocking=True),
            xy=payload["prompt_xy"][offset:offset + 1].to(device, non_blocking=True),
            area=payload["prompt_area"][offset:offset + 1].to(device, non_blocking=True),
            active=payload["prompt_active"][offset:offset + 1].to(device, non_blocking=True),
        )
