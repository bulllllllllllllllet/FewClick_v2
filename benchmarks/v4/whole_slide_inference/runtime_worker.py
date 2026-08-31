"""Line-oriented persistent FewClick_j5 runtime.

The process is deliberately single-threaded: it owns one GPU and serializes all
requests, while retaining formal J5 weights and region-cache stores in memory.
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd
import torch

from benchmarks.v4.whole_slide_inference import infer_wsi
from benchmarks.v4.whole_slide_inference.src.local_cache import (
    LocalizedPreparationCache,
    materialize_preparation_cache,
)


class Runtime:
    def __init__(self) -> None:
        self.models: dict[tuple[str, ...], tuple[object, dict]] = {}
        self.caches: dict[tuple[str, int], object] = {}
        self.localized: dict[tuple[str, ...], LocalizedPreparationCache] = {}

    @staticmethod
    def _artifact_key(args: object) -> tuple[str, ...]:
        return tuple(str(getattr(args, key)) for key in (
            "tile_index", "cell_feature_manifest", "region_cache_manifest",
            "config", "phase2_config", "phase5_config", "phase2_checkpoint",
            "cell_checkpoint", "phase5_checkpoint", "joint_checkpoint",
        ))

    def _localize(self, args: object) -> dict:
        if args.region_cache_manifest is None or args.local_cache_root is None:
            return {"enabled": False, "hit": False, "seconds": 0.0}
        started_at = time.perf_counter()
        artifact_key = self._artifact_key(args)
        memory_hit = artifact_key in self.localized
        if memory_hit:
            localized = self.localized[artifact_key]
        else:
            localized = materialize_preparation_cache(
                tile_index=args.tile_index,
                cell_feature_manifest=args.cell_feature_manifest,
                region_cache_manifest=args.region_cache_manifest,
                cache_root=args.local_cache_root,
                max_bytes=args.local_cache_max_bytes,
                min_free_bytes=args.local_cache_min_free_bytes,
                cache_key_parts=artifact_key[3:],
            )
            self.localized[artifact_key] = localized
        args.tile_index = localized.tile_index
        args.cell_feature_manifest = localized.cell_feature_manifest
        args.region_cache_manifest = localized.region_cache_manifest
        report = {
            "enabled": True,
            "hit": bool(memory_hit or localized.cache_hit),
            "memory_hit": memory_hit,
            "seconds": time.perf_counter() - started_at,
            "bytes_on_disk": localized.bytes_on_disk,
            "cache_dir": str(localized.cache_dir),
        }
        args._local_cache_materialized = True
        args._local_cache_report = report
        return report

    def _device(self, args: object) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("whole-slide J5 inference requires CUDA")
        torch.cuda.set_device(args.gpus[0])
        return torch.device("cuda", args.gpus[0])

    def warm(self, args: object) -> dict:
        local_cache = self._localize(args)
        device = self._device(args)
        model_key = tuple(str(getattr(args, key)) for key in (
            "config", "phase2_config", "phase5_config", "phase1_checkpoint", "phase2_checkpoint",
            "cell_checkpoint", "phase5_checkpoint", "joint_checkpoint",
        )) + (str(device),)
        if model_key not in self.models:
            self.models[model_key] = infer_wsi.load_model(args, device)
        if args.region_cache_manifest is not None:
            cache_key = (str(args.region_cache_manifest), int(args.region_cache_max_open_shards))
            if cache_key not in self.caches:
                # The adapter serializes one active preparation at a time. Drop
                # the previous slide's tensors before retaining the new bundle.
                self.caches.clear()
                expected_rows = len(pd.read_parquet(args.tile_index))
                cache = infer_wsi.RegionCacheStore(
                    args.region_cache_manifest, expected_rows, args.region_cache_max_open_shards
                )
                cache.preload()
                self.caches[cache_key] = cache
            resident_shards = len(self.caches[cache_key].cache)
        else:
            resident_shards = 0
        return {
            "event": "runtime_ready", "device": str(device),
            "local_cache": local_cache, "resident_region_shards": resident_shards,
        }

    def run(self, args: object) -> None:
        self.warm(args)
        original_model = infer_wsi.load_model
        original_cache = infer_wsi.RegionCacheStore
        device = self._device(args)
        model_key = tuple(str(getattr(args, key)) for key in (
            "config", "phase2_config", "phase5_config", "phase1_checkpoint", "phase2_checkpoint",
            "cell_checkpoint", "phase5_checkpoint", "joint_checkpoint",
        )) + (str(device),)
        infer_wsi.load_model = lambda _args, _device: self.models[model_key]  # type: ignore[assignment]
        if args.region_cache_manifest is not None:
            cache_key = (str(args.region_cache_manifest), int(args.region_cache_max_open_shards))
            infer_wsi.RegionCacheStore = lambda _path, _count, _max: self.caches[cache_key]  # type: ignore[assignment]
        try:
            infer_wsi.main(args)
        finally:
            infer_wsi.load_model = original_model  # type: ignore[assignment]
            infer_wsi.RegionCacheStore = original_cache  # type: ignore[assignment]


def emit(value: dict) -> None:
    print(json.dumps(value), flush=True)


def main() -> None:
    runtime = Runtime()
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            request_id = request.get("request_id", "unknown")
            if request["op"] == "shutdown":
                emit({"event": "runtime_stopped", "request_id": request_id})
                return
            args = infer_wsi.parse_args(request["argv"])
            if request["op"] == "warm":
                emit({**runtime.warm(args), "request_id": request_id})
            elif request["op"] == "run":
                runtime.run(args)
                emit({"event": "runtime_run_complete", "request_id": request_id})
            else:
                raise ValueError("unknown runtime operation")
        except Exception as exc:
            emit({"event": "runtime_error", "error": repr(exc), "request_id": request.get("request_id", "unknown")})


if __name__ == "__main__":
    main()
