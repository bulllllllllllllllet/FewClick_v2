#!/usr/bin/env python3
"""Materialize exact prompt-independent J5 regions for a complete WSI index."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.whole_slide_inference.infer_wsi import SlideReader, load_model, tensor_batch
from benchmarks.v4.whole_slide_inference.src.features import CellFeatureStore
from benchmarks.v4.whole_slide_inference.src.prompt_transfer import encode_regions
from benchmarks.v4.whole_slide_inference.src.region_cache import sha256
from benchmarks.v4.whole_slide_inference.src.tiling import validate_tile_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "phase2-config", "phase5-config", "phase2-checkpoint", "cell-checkpoint", "phase5-checkpoint", "joint-checkpoint", "tile-index", "cell-feature-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shard-rows", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gpus) != 1 or not torch.cuda.is_available():
        raise RuntimeError("region cache build requires exactly one CUDA GPU")
    if args.batch_size < 1 or args.shard_rows < 1:
        raise ValueError("batch-size and shard-rows must be positive")
    torch.cuda.set_device(args.gpus[0])
    device = torch.device("cuda", args.gpus[0])
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"region_cache_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    rows = pd.read_parquet(args.tile_index).sort_values("source_index").to_dict("records")
    validate_tile_rows(rows)
    reader = SlideReader(Path(rows[0]["wsi_path"]))
    store = CellFeatureStore(args.cell_feature_manifest, len(rows))
    model, load_report = load_model(args, device)
    max_cells = int(load_config(args.config)["data"]["max_cells_per_patch"])
    shard_rows: list[dict[str, torch.Tensor]] = []
    manifest_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        images, cells = tensor_batch(batch_rows, reader, store, max_cells, args.num_workers, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # The formal WSI path encodes decoder tiles in --batch-size batches,
            # but encodes each prompt tile individually.  Preserve both contracts:
            # the cell path is batch-sensitive, so reusing the former for prompts
            # is observably not bitwise equivalent.
            encoded = encode_regions(model, images, **cells)
            prompt_encoded = [
                encode_regions(
                    model,
                    images[index:index + 1],
                    **{key: value[index:index + 1] for key, value in cells.items()},
                )
                for index in range(len(batch_rows))
            ]
        assert encoded.assignment_low is not None
        for index, row in enumerate(batch_rows):
            shard_rows.append({
                "source_index": torch.tensor(int(row["source_index"]), dtype=torch.int64),
                "assignment_low": encoded.assignment_low[index].detach().cpu(),
                "tokens": encoded.tokens[index].detach().cpu(),
                "xy": encoded.xy[index].detach().cpu(),
                "area": encoded.area[index].detach().cpu(),
                "active": encoded.active[index].detach().cpu(),
                "prompt_tokens": prompt_encoded[index].tokens[0].detach().cpu(),
                "prompt_xy": prompt_encoded[index].xy[0].detach().cpu(),
                "prompt_area": prompt_encoded[index].area[0].detach().cpu(),
                "prompt_active": prompt_encoded[index].active[0].detach().cpu(),
            })
        if len(shard_rows) >= args.shard_rows or start + len(batch_rows) == len(rows):
            first = int(shard_rows[0]["source_index"])
            end = first + len(shard_rows)
            payload = {key: torch.stack([item[key] for item in shard_rows]) for key in shard_rows[0]}
            shard_path = output / f"shard_{first:06d}_{end:06d}.pt"
            torch.save(payload, shard_path)
            manifest_rows.append({"shard_path": str(shard_path), "start": first, "end": end, "rows": len(shard_rows), "sha256": sha256(shard_path)})
            shard_rows = []
        print(json.dumps({"event": "regions_cached", "done": start + len(batch_rows), "total": len(rows)}), flush=True)
    manifest_path = output / "region_cache_index.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    metadata = {"status": "complete", "timestamp": stamp, "tile_count": len(rows), "manifest": str(manifest_path), "load_report": load_report, "cache_contract": "decoder assignment_low/tokens/xy/area/active at formal --batch-size; prompt tokens/xy/area/active at formal batch=1; full assignment rebuilt with formal bilinear+renormalize"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
