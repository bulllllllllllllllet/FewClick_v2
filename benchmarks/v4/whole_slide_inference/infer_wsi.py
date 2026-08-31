#!/usr/bin/env python3
"""Prompt-conditioned whole-slide inference with overlap-weighted stitching."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VIPS_CONCURRENCY", "1")
import pyvips
import torch
from PIL import Image
from vipa_slide_reader import get_slide

pyvips.cache_set_max(0)
pyvips.cache_set_max_mem(64 * 1024 * 1024)

from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_3_cell_region.src.cells import collate_cells
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import SIZE_NAMES
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import JointPromptMaskModel, load_joint_components
from benchmarks.v4.whole_slide_inference.src.features import CellFeatureStore
from benchmarks.v4.whole_slide_inference.src.local_cache import materialize_preparation_cache
from benchmarks.v4.whole_slide_inference.src.prompt_transfer import (
    EncodedRegions,
    decode_regions_with_task,
    encode_regions,
)
from benchmarks.v4.whole_slide_inference.src.region_cache import RegionCacheStore
from benchmarks.v4.whole_slide_inference.src.tiling import (
    OverlapAccumulator,
    blend_window,
    slide_dimensions_from_rows,
    validate_tile_rows,
)


# This module stays loaded inside the persistent runtime. Cache only after a
# stat-based identity check, so every run still records a digest for the exact
# immutable file it used and changes are never silently ignored.
_SHA256_CACHE: dict[Path, tuple[tuple[int, int, int, int], str]] = {}
_TILE_ROWS_CACHE: dict[Path, tuple[tuple[int, int, int, int], list[dict]]] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--phase5-config", type=Path, required=True)
    parser.add_argument("--phase1-checkpoint", type=Path)
    parser.add_argument("--phase2-checkpoint", type=Path, required=True)
    parser.add_argument("--cell-checkpoint", type=Path, required=True)
    parser.add_argument("--phase5-checkpoint", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--tile-index", type=Path, required=True)
    parser.add_argument("--cell-feature-manifest", type=Path, required=True)
    parser.add_argument("--region-cache-manifest", type=Path)
    parser.add_argument("--local-cache-root", type=Path)
    parser.add_argument("--local-cache-max-bytes", type=int, default=300 * 1024**3)
    parser.add_argument("--local-cache-min-free-bytes", type=int, default=500 * 1024**3)
    parser.add_argument("--region-cache-max-open-shards", type=int, default=64)
    parser.add_argument("--prompt-json", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/nfs-medical3/zyh/v4/whole_slide_inference"),
    )
    parser.add_argument("--timestamp")
    parser.add_argument(
        "--scratch-root", type=Path,
        help="Local temporary directory for overlap memmaps; final outputs remain under --output-root.",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    # Keep the formal CLI's complete audit outputs by default. Production can
    # opt out of the two large derived TIFFs while always retaining the binary
    # mask, which remains the canonical J5 result.
    parser.add_argument("--write-probability-tiff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-coverage-tiff", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def sha256(path: Path) -> str:
    path = Path(path)
    identity = _file_identity(path)
    cached = _SHA256_CACHE.get(path)
    if cached is not None and cached[0] == identity:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA256_CACHE[path] = (identity, value)
    return value


def load_tile_rows(path: Path) -> list[dict]:
    path = Path(path)
    identity = _file_identity(path)
    cached = _TILE_ROWS_CACHE.get(path)
    if cached is not None and cached[0] == identity:
        return cached[1]
    rows = pd.read_parquet(path).sort_values("source_index").to_dict("records")
    _TILE_ROWS_CACHE[path] = (identity, rows)
    return rows


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


class SlideReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        # Keep WSI decoding on the same OpenSlide/KFB implementation as the
        # HE-only feature extractor.  In particular, the packaged KFB reader
        # owns an isolated libjpeg9 and must not fall back to CellAtlas's
        # legacy module.KFBreader binary.
        self.image = get_slide(str(path))
        self.width, self.height = self.image.dimensions

    def read(self, row: dict) -> np.ndarray:
        x = int(row["x_level0"]); y = int(row["y_level0"])
        width = int(row["width_level0"]); height = int(row["height_level0"])
        output_width = int(row["width_10x"]); output_height = int(row["height_10x"])
        if output_width != output_height or width != height:
            raise ValueError("J5 expects square model tiles")
        patch = self.image.read_region((x, y), 0, (width, height)).convert("RGB")
        return np.asarray(
            patch.resize((output_width, output_height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[JointPromptMaskModel, dict]:
    cfg = load_config(args.config); p2cfg = load_config(args.phase2_config); p5cfg = load_config(args.phase5_config)
    phase1_checkpoint = getattr(args, "phase1_checkpoint", None)
    if phase1_checkpoint is not None:
        p2cfg["model"] = {**p2cfg["model"], "pretrained_weights": str(phase1_checkpoint)}
    if bool(cfg["model"].get("train_parent_context", False)):
        raise ValueError("the deployed whole-slide entry point is for the formal J5 checkpoint, not Phase4/J10")
    phase2, cell, prompt, upstream = load_joint_components(
        p2cfg, args.phase2_checkpoint, args.cell_checkpoint, args.phase5_checkpoint,
        p5cfg, int(cfg["data"]["cell_feature_dim"]),
    )
    mc = cfg["model"]
    model = JointPromptMaskModel(
        phase2, cell, prompt, int(mc["region_dim"]), int(mc["graph_heads"]),
        int(mc["graph_layers"]), int(mc["graph_neighbours"]), float(mc["graph_dropout"]),
        float(mc["residual_limit"]),
        train_phase2_embedding=bool(mc.get("train_phase2_embedding", False)),
        train_phase2_assignment=bool(mc.get("train_phase2_assignment", False)),
        train_cell=bool(mc.get("train_cell", False)),
        train_prompt=bool(mc.get("train_prompt", False)),
        train_decoder=bool(mc.get("train_decoder", True)),
        train_parent_context=False,
        train_backbone_layer4=bool(mc.get("train_backbone_layer4", False)),
        num_classes=len(p2cfg["data"]["class_map"]),
        ignore_index=int(cfg["data"]["ignore_index"]),
    )
    payload = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"J5 checkpoint mismatch missing={missing} unexpected={unexpected}")
    report = {
        "upstream": upstream,
        "joint": {"checkpoint": str(args.joint_checkpoint), "epoch": int(payload["epoch"]),
                  "missing": list(missing), "unexpected": list(unexpected)},
    }
    return model.to(device).eval(), report


def load_prompts(path: Path, width_level0: int, height_level0: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("coordinate_space") != "level0":
        raise ValueError("prompt_json.coordinate_space must be exactly 'level0'")
    size = str(value.get("prompt_size"))
    if size not in SIZE_NAMES:
        raise ValueError(f"prompt_size must be one of {SIZE_NAMES}")
    output = {"coordinate_space": "level0", "prompt_size": size}
    for sign in ("positive", "negative"):
        points = value.get(sign)
        if not isinstance(points, list) or not points:
            raise ValueError(f"prompt_json.{sign} must be a non-empty list")
        parsed = []
        for order, point in enumerate(points, start=1):
            x = float(point["x"]); y = float(point["y"])
            if not (0 <= x < width_level0 and 0 <= y < height_level0):
                raise ValueError(f"{sign} prompt {order} lies outside the WSI")
            parsed.append({"x": x, "y": y})
        output[sign] = parsed
    return output


def choose_prompt_tile(rows: list[dict], point: dict) -> dict:
    x = float(point["x"]); y = float(point["y"])
    candidates = [
        row for row in rows
        if int(row["x_level0"]) <= x < int(row["x_level0"]) + int(row["width_level0"])
        and int(row["y_level0"]) <= y < int(row["y_level0"]) + int(row["height_level0"])
    ]
    if not candidates:
        raise RuntimeError(f"prompt {(x, y)} is not covered by the tile index")
    return min(candidates, key=lambda row: (
        (x - (int(row["x_level0"]) + int(row["width_level0"]) / 2)) ** 2
        + (y - (int(row["y_level0"]) + int(row["height_level0"]) / 2)) ** 2,
        int(row["source_index"]),
    ))


def tensor_batch(
    rows: list[dict], reader: SlideReader, store: CellFeatureStore,
    max_cells: int, workers: int, device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(reader.read, rows))
    else:
        images = [reader.read(row) for row in rows]
    cell_items = []
    for row in rows:
        cells, total = store.get(int(row["source_index"]), str(row["patch_id"]))
        cell_items.append((str(row["patch_id"]), cells, total))
    packed = collate_cells(cell_items, max_cells=max_cells)
    image_tensor = torch.stack([he_to_tensor(image) for image in images]).to(device, non_blocking=True)
    packed = {key: value.to(device, non_blocking=True) for key, value in packed.items()}
    return image_tensor, packed


def subset_encoded(encoded: EncodedRegions, index: int) -> EncodedRegions:
    return EncodedRegions(*(
        value[index:index + 1]
        for value in (encoded.assignment, encoded.tokens, encoded.xy, encoded.area, encoded.active)
    ))


@torch.no_grad()
def encode_global_prompt_task(
    model: JointPromptMaskModel,
    rows: list[dict],
    prompts: dict,
    reader: SlideReader,
    store: CellFeatureStore,
    max_cells: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor] | None, list[dict], list[dict]]:
    selected = []
    for sign in ("positive", "negative"):
        for order, point in enumerate(prompts[sign], start=1):
            selected.append((sign, order, point, choose_prompt_tile(rows, point)))
    unique_rows = {int(item[3]["source_index"]): item[3] for item in selected}
    encoded_by_source = {}
    for source_index, row in unique_rows.items():
        images, cells = tensor_batch([row], reader, store, max_cells, 0, device)
        with torch.autocast("cuda", dtype=amp_dtype):
            encoded_by_source[source_index] = encode_regions(model, images, **cells)
    records = {"positive": [], "negative": []}; keys = {"positive": set(), "negative": set()}
    tokens = {"positive": [], "negative": []}; coordinates = {"positive": [], "negative": []}
    for sign, order, point, row in selected:
        encoded = encoded_by_source[int(row["source_index"])]
        local = torch.tensor([[
            (float(point["x"]) - int(row["x_level0"])) / int(row["width_level0"]),
            (float(point["y"]) - int(row["y_level0"])) / int(row["height_level0"]),
        ]], device=device)
        slot = int(torch.cdist(local.float(), encoded.xy[0].float()).argmin())
        key = (int(row["source_index"]), slot)
        record = {
            "sign": sign, "order": order,
            "x_level0": float(point["x"]), "y_level0": float(point["y"]),
            "patch_id": str(row["patch_id"]), "source_index": int(row["source_index"]),
            "region_slot": slot,
            "region_centroid_xy_local": [float(value) for value in encoded.xy[0, slot].float().cpu()],
        }
        records[sign].append(record)
        if key not in keys[sign]:
            keys[sign].add(key)
            tokens[sign].append(encoded.tokens[0, slot])
            coordinates[sign].append(encoded.xy[0, slot])
    conflicts = keys["positive"] & keys["negative"]
    if conflicts:
        conflict_records = [
            {"source_index": source_index, "region_slot": slot}
            for source_index, slot in sorted(conflicts)
        ]
        return None, records["positive"] + records["negative"], conflict_records
    prompt_batch = {
        "positive_tokens": torch.stack(tokens["positive"])[None],
        "negative_tokens": torch.stack(tokens["negative"])[None],
        "positive_xy": torch.stack(coordinates["positive"])[None],
        "negative_xy": torch.stack(coordinates["negative"])[None],
        "positive_mask": torch.ones(1, len(tokens["positive"]), dtype=torch.bool, device=device),
        "negative_mask": torch.ones(1, len(tokens["negative"]), dtype=torch.bool, device=device),
        "prompt_size_id": torch.tensor([SIZE_NAMES.index(prompts["prompt_size"])], device=device),
    }
    with torch.autocast("cuda", dtype=amp_dtype):
        task = model.decoder.prompt_model.encode_prompt_task(prompt_batch)
    return task, records["positive"] + records["negative"], []


@torch.no_grad()
def encode_global_prompt_task_from_cache(
    model: JointPromptMaskModel,
    rows: list[dict],
    prompts: dict,
    cache: RegionCacheStore,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor] | None, list[dict], list[dict]]:
    selected = []
    for sign in ("positive", "negative"):
        for order, point in enumerate(prompts[sign], start=1):
            selected.append((sign, order, point, choose_prompt_tile(rows, point)))
    unique_rows = {int(item[3]["source_index"]): item[3] for item in selected}
    encoded_by_source = {
        source_index: cache.prompt_encoded(row, device)
        for source_index, row in unique_rows.items()
    }
    records = {"positive": [], "negative": []}; keys = {"positive": set(), "negative": set()}
    tokens = {"positive": [], "negative": []}; coordinates = {"positive": [], "negative": []}
    for sign, order, point, row in selected:
        encoded = encoded_by_source[int(row["source_index"])]
        local = torch.tensor([[
            (float(point["x"]) - int(row["x_level0"])) / int(row["width_level0"]),
            (float(point["y"]) - int(row["y_level0"])) / int(row["height_level0"]),
        ]], device=device)
        slot = int(torch.cdist(local.float(), encoded.xy[0].float()).argmin())
        key = (int(row["source_index"]), slot)
        records[sign].append({
            "sign": sign, "order": order, "x_level0": float(point["x"]), "y_level0": float(point["y"]),
            "patch_id": str(row["patch_id"]), "source_index": int(row["source_index"]),
            "region_slot": slot,
            "region_centroid_xy_local": [float(value) for value in encoded.xy[0, slot].float().cpu()],
        })
        if key not in keys[sign]:
            keys[sign].add(key); tokens[sign].append(encoded.tokens[0, slot]); coordinates[sign].append(encoded.xy[0, slot])
    conflicts = keys["positive"] & keys["negative"]
    if conflicts:
        return None, records["positive"] + records["negative"], [
            {"source_index": source_index, "region_slot": slot}
            for source_index, slot in sorted(conflicts)
        ]
    prompt_batch = {
        "positive_tokens": torch.stack(tokens["positive"])[None], "negative_tokens": torch.stack(tokens["negative"])[None],
        "positive_xy": torch.stack(coordinates["positive"])[None], "negative_xy": torch.stack(coordinates["negative"])[None],
        "positive_mask": torch.ones(1, len(tokens["positive"]), dtype=torch.bool, device=device),
        "negative_mask": torch.ones(1, len(tokens["negative"]), dtype=torch.bool, device=device),
        "prompt_size_id": torch.tensor([SIZE_NAMES.index(prompts["prompt_size"])], device=device),
    }
    with torch.autocast("cuda", dtype=amp_dtype):
        task = model.decoder.prompt_model.encode_prompt_task(prompt_batch)
    return task, records["positive"] + records["negative"], []


def write_pyramid(path: Path, array: np.ndarray, pixel_format: str) -> None:
    image = pyvips.Image.new_from_memory(
        memoryview(array), int(array.shape[1]), int(array.shape[0]), 1, pixel_format
    )
    tile_width = min(512, max(16, (int(array.shape[1]) // 16) * 16))
    tile_height = min(512, max(16, (int(array.shape[0]) // 16) * 16))
    image.tiffsave(
        str(path), tile=True, tile_width=tile_width, tile_height=tile_height, pyramid=True,
        bigtiff=True, compression="deflate",
    )


def main(args: argparse.Namespace | None = None) -> None:
    run_started_at = time.perf_counter()
    args = args or parse_args()
    local_cache_started_at = time.perf_counter()
    if (
        args.region_cache_manifest is not None
        and args.local_cache_root is not None
        and not getattr(args, "_local_cache_materialized", False)
    ):
        cache_key_parts = tuple(str(getattr(args, key)) for key in (
            "config", "phase2_config", "phase5_config", "phase2_checkpoint",
            "cell_checkpoint", "phase5_checkpoint", "joint_checkpoint",
        ))
        localized = materialize_preparation_cache(
            tile_index=args.tile_index,
            cell_feature_manifest=args.cell_feature_manifest,
            region_cache_manifest=args.region_cache_manifest,
            cache_root=args.local_cache_root,
            max_bytes=args.local_cache_max_bytes,
            min_free_bytes=args.local_cache_min_free_bytes,
            cache_key_parts=cache_key_parts,
        )
        args.tile_index = localized.tile_index
        args.cell_feature_manifest = localized.cell_feature_manifest
        args.region_cache_manifest = localized.region_cache_manifest
        args._local_cache_report = {
            "enabled": True, "hit": localized.cache_hit,
            "bytes_on_disk": localized.bytes_on_disk,
            "cache_dir": str(localized.cache_dir),
        }
        args._local_cache_materialized = True
    if len(args.gpus) != 1:
        raise ValueError("one WSI run currently requires exactly one --gpus ID")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("whole-slide J5 inference requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpus[0]); device = torch.device("cuda", args.gpus[0])
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    stamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"wsi_inference_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    timing_seconds: dict[str, float] = {
        "localize_preparation_cache": time.perf_counter() - local_cache_started_at,
    }
    stage_started_at = time.perf_counter()
    rows = load_tile_rows(args.tile_index)
    timing_seconds["load_tile_index"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    width, height, downsample = validate_tile_rows(rows)
    source_width_level0, source_height_level0 = slide_dimensions_from_rows(rows)
    timing_seconds["validate_tile_index"] = time.perf_counter() - stage_started_at
    wsi_path = Path(rows[0]["wsi_path"])
    reader = None
    stage_started_at = time.perf_counter()
    if args.region_cache_manifest is None:
        reader = SlideReader(wsi_path)
        if (reader.width, reader.height) != (source_width_level0, source_height_level0):
            raise RuntimeError(
                "tile index level0 dimensions do not match the source WSI: "
                f"index={(source_width_level0, source_height_level0)} "
                f"source={(reader.width, reader.height)}"
            )
    timing_seconds["open_source_wsi"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    prompts = load_prompts(args.prompt_json, source_width_level0, source_height_level0)
    timing_seconds["load_and_validate_prompt"] = time.perf_counter() - stage_started_at
    inputs = [
        args.config, args.phase2_config, args.phase5_config, args.phase2_checkpoint,
        args.cell_checkpoint, args.phase5_checkpoint, args.joint_checkpoint,
        args.tile_index, args.cell_feature_manifest, args.prompt_json,
    ]
    phase1_checkpoint = getattr(args, "phase1_checkpoint", None)
    if phase1_checkpoint is not None:
        inputs.append(phase1_checkpoint)
    if args.region_cache_manifest is not None:
        inputs.append(args.region_cache_manifest)
    stage_started_at = time.perf_counter()
    input_hashes = {str(path): sha256(path) for path in inputs}
    timing_seconds["hash_inputs"] = time.perf_counter() - stage_started_at
    run_manifest = {
        "timestamp": stamp, "status": "started", "split": str(rows[0]["split"]),
        "test_used": str(rows[0]["split"]) == "test",
        "wsi_id": str(rows[0]["wsi_id"]), "wsi_path": str(wsi_path),
        "tile_count": len(rows), "output_shape_10x": [height, width],
        "level0_downsample": downsample, "gpus": args.gpus,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "seed": args.seed, "threshold": args.threshold, "amp_dtype": args.amp_dtype,
        "local_cache": getattr(args, "_local_cache_report", {"enabled": False, "hit": False}),
        "inputs": input_hashes,
    }
    save_json(output / "run_manifest.json", run_manifest)
    stage_started_at = time.perf_counter()
    store = None if args.region_cache_manifest is not None else CellFeatureStore(args.cell_feature_manifest, len(rows))
    region_cache = None if args.region_cache_manifest is None else RegionCacheStore(
        args.region_cache_manifest, len(rows), args.region_cache_max_open_shards
    )
    timing_seconds["initialize_input_stores"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    model, load_report = load_model(args, device)
    timing_seconds["load_model"] = time.perf_counter() - stage_started_at
    max_cells = int(load_config(args.config)["data"]["max_cells_per_patch"])
    stage_started_at = time.perf_counter()
    if region_cache is None:
        assert store is not None and reader is not None
        prompt_task, prompt_records, conflicts = encode_global_prompt_task(model, rows, prompts, reader, store, max_cells, device, amp_dtype)
    else:
        prompt_task, prompt_records, conflicts = encode_global_prompt_task_from_cache(model, rows, prompts, region_cache, device, amp_dtype)
    timing_seconds["encode_prompt_task"] = time.perf_counter() - stage_started_at
    timing_seconds["prepare_inputs_and_prompt"] = time.perf_counter() - run_started_at
    if conflicts:
        metadata = {
            **run_manifest, "status": "abstained_prompt_conflict", "mask_returned": False,
            "prompt_records": prompt_records, "conflicts": conflicts, "load_report": load_report,
            "timing_seconds": {**timing_seconds, "total": time.perf_counter() - run_started_at},
        }
        save_json(output / "metadata.json", metadata)
        print(json.dumps(metadata, indent=2), flush=True)
        return
    accumulator_started_at = time.perf_counter()
    scratch_root = output if args.scratch_root is None else args.scratch_root / f"wsi_inference_{stamp}"
    scratch_root.mkdir(parents=True, exist_ok=False)
    accumulator = OverlapAccumulator(scratch_root, height, width, args.threshold)
    timing_seconds["initialize_overlap_accumulators"] = time.perf_counter() - accumulator_started_at
    tile_height = int(rows[0]["height_10x"]); tile_width = int(rows[0]["width_10x"])
    window = blend_window(tile_height, tile_width)
    completed_path = output / "completed.jsonl"; failures_path = output / "failures.jsonl"
    decode_started_at = time.perf_counter()
    with completed_path.open("a", encoding="utf-8") as completed, failures_path.open("a", encoding="utf-8") as failures:
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
                    if region_cache is None:
                        assert store is not None and reader is not None
                        images, cells = tensor_batch(batch_rows, reader, store, max_cells, args.num_workers, device)
                        encoded = encode_regions(model, images, **cells)
                    else:
                        encoded = region_cache.encoded(batch_rows, device)
                    result = decode_regions_with_task(model, encoded, prompt_task)
                probabilities = result["pixel_probability"].float().cpu().numpy()
                for row, probability in zip(batch_rows, probabilities, strict=True):
                    accumulator.add(probability, int(row["x_10x"]), int(row["y_10x"]), window)
                    completed.write(json.dumps({
                        "source_index": int(row["source_index"]), "patch_id": str(row["patch_id"]),
                        "probability_min": float(probability.min()),
                        "probability_max": float(probability.max()),
                    }) + "\n")
                completed.flush()
            except Exception as exc:
                failures.write(json.dumps({"start": start, "error": repr(exc)}) + "\n")
                failures.flush()
                raise
            print({"event": "tiles_complete", "done": start + len(batch_rows), "total": len(rows)}, flush=True)
    timing_seconds["tile_decode_and_overlap_accumulation"] = time.perf_counter() - decode_started_at
    finalize_started_at = time.perf_counter()
    probability, mask, probability_range = accumulator.finalize(
        write_probability=args.write_probability_tiff,
        return_probability_range=True,
    )
    timing_seconds["finalize_probability_and_mask"] = time.perf_counter() - finalize_started_at
    probability_tiff = output / "probability_10x_pyramid.tif"
    mask_tiff = output / "mask_10x_pyramid.tif"
    coverage_tiff = output / "blend_weight_10x_pyramid.tif"
    output_write_started_at = time.perf_counter()
    if args.write_probability_tiff:
        write_pyramid(probability_tiff, probability, "float")
    write_pyramid(mask_tiff, mask, "uchar")
    if args.write_coverage_tiff:
        write_pyramid(coverage_tiff, accumulator.weight_sum, "float")
    timing_seconds["write_pyramid_outputs"] = time.perf_counter() - output_write_started_at
    timing_seconds["total"] = time.perf_counter() - run_started_at
    metadata = {
        **run_manifest, "status": "complete", "mask_returned": True,
        "prompt_records": prompt_records, "conflicts": [], "load_report": load_report,
        "probability_tiff": str(probability_tiff) if args.write_probability_tiff else None,
        "mask_tiff": str(mask_tiff),
        "coverage_tiff": str(coverage_tiff) if args.write_coverage_tiff else None,
        "probability_range": list(probability_range),
        "positive_fraction_at_threshold": float((mask > 0).mean()),
        "coordinate_contract": {
            "prompt_input": "level0 pixels", "output": "10x pixels",
            "level0_to_output": f"floor(level0 / {downsample})",
        },
        "scope": "one complete WSI; formal J5 without Phase4 parent context",
        "timing_seconds": timing_seconds,
    }
    save_json(output / "metadata.json", metadata)
    if scratch_root != output:
        accumulator.cleanup()
        scratch_root.rmdir()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
