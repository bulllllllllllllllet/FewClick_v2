#!/usr/bin/env python3
"""Run frozen v4 J5 inference with the offline cell input deterministically zeroed.

This is a downstream ablation adapter, not a v4 model change.  It retains the
original feature-store lookup first, so shard coverage and patch identities are
checked exactly as in the full-cell arm.  It then replaces every per-patch cell
matrix by a correctly shaped empty matrix and sets its total cell count to zero.
The frozen Phase 3 module consequently receives zero cell evidence while every
other input, checkpoint, prompt, and tile remains unchanged.
"""
from __future__ import annotations

import argparse
import errno
import json
import sys
from pathlib import Path

import numpy as np

from benchmarks.v4.whole_slide_inference import infer_wsi as v4_inference


CELL_FEATURE_DIM = 74  # [x, y, log(area)] + 7-way class one-hot + reg-64


class ZeroCellFeatureStore(v4_inference.CellFeatureStore):
    """Validate source cache entries, then expose no cell tokens to J5."""

    def get(self, source_index: int, patch_id: str) -> tuple[np.ndarray, int]:
        cells, _ = super().get(source_index, patch_id)
        if cells.shape[1] != CELL_FEATURE_DIM:
            raise ValueError(
                f"{patch_id}: expected {CELL_FEATURE_DIM} cell features, got {cells.shape}"
            )
        return np.empty((0, CELL_FEATURE_DIM), dtype=np.float32), 0


def parse_wrapper_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ablation-label", default="cell_zeroed",
        help="Recorded only in the v5 ablation contract; does not alter v4 inference.",
    )
    parser.add_argument("--help", action="store_true")
    known, remaining = parser.parse_known_args(argv)
    if known.help:
        parser.print_help()
        v4_inference.parse_args(["--help"])
    return known, remaining


def main(argv: list[str] | None = None) -> None:
    wrapper, remaining = parse_wrapper_args(argv)
    original_store = v4_inference.CellFeatureStore
    v4_inference.CellFeatureStore = ZeroCellFeatureStore
    try:
        # Keep v4's own parser and all of its strict runtime validation intact.
        original_argv = sys.argv
        sys.argv = [original_argv[0], *remaining]
        args = v4_inference.parse_args()
        # v4 creates ``output`` before constructing its overlap accumulator;
        # its default scratch path is the already-created output directory.
        # Give this v5 adapter a sibling scratch parent without changing v4.
        if args.scratch_root is None:
            args.scratch_root = args.output_root / "_scratch"
        timestamp = args.timestamp
        output = args.output_root / f"wsi_inference_{timestamp}"
        try:
            v4_inference.main(args)
        except OSError as exc:
            # The frozen entry point writes its complete metadata before
            # deleting memmaps, then may race with NFS directory cleanup.
            # Treat only that post-output cleanup error as non-fatal.
            completed_metadata = output / "metadata.json"
            if exc.errno != errno.ENOTEMPTY or not completed_metadata.is_file():
                raise
            metadata = json.loads(completed_metadata.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete" or not metadata.get("mask_returned"):
                raise
    finally:
        v4_inference.CellFeatureStore = original_store
        sys.argv = original_argv
    contract = {
        "ablation_label": wrapper.ablation_label,
        "cell_input": "zeroed_after_strict_feature_cache_identity_validation",
        "cell_feature_dim": CELL_FEATURE_DIM,
        "baseline_module": "benchmarks.v4.whole_slide_inference.infer_wsi",
        "inference_output": str(output),
    }
    (output / "v5_cell_ablation_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
