#!/usr/bin/env python3
"""Explicit held-out evaluation or prompt-independent cache export."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_3_cell_region.src.semantic_cell_region import SemanticCellToRegion
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import SemanticDataset, sha256_file
from benchmarks.v5.phase_3_cell_region.src.semantic_training import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mode", choices=("evaluate", "export"), required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp must be MMDD_HHMMSS")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use FewClick NFS artifact root")
    torch.set_num_threads(4)
    dataset = SemanticDataset(args.manifest, args.split)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["schema_version"] != "semantic_phase3_v1" or saved["manifest_sha256"] != dataset.manifest_hash:
        raise ValueError("checkpoint/manifest fingerprint mismatch")
    for key in ("identity_classes", "tissue_classes"):
        if saved[key] != dataset.spec[key]:
            raise ValueError(f"checkpoint {key} mismatch")
    model = SemanticCellToRegion(len(saved["identity_classes"]), len(saved["tissue_classes"])).to(args.device)
    model.load_state_dict(saved["model"], strict=True)
    model.set_stage(saved["stage"])
    model.eval()
    if args.mode == "export" and saved["stage"] != "fusion":
        raise ValueError("region cache requires a trained fusion checkpoint")
    output = args.output_root / f"{args.mode}_{args.split}_{args.timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": saved["schema_version"], "checkpoint_sha256": sha256_file(args.checkpoint),
                "manifest_sha256": dataset.manifest_hash, "split": args.split,
                "identity_classes": saved["identity_classes"], "tissue_classes": saved["tissue_classes"],
                "provenance": dataset.spec["provenance"], "cache_scope": "patch region tokens; whole-slide routing adapter not yet implemented"}
    if args.mode == "evaluate":
        metadata["metrics"] = evaluate(model, dataset, torch.device(args.device))
    else:
        rows = []
        with torch.no_grad():
            for index, item in enumerate(dataset):
                inputs = {key: value.to(args.device) for key, value in item["inputs"].items()}
                # No target fields are passed into the model or exported.
                result = model(**inputs)
                if not torch.isfinite(result["fused_tokens"]).all():
                    raise RuntimeError("non-finite region cache")
                path = output / f"patch_{index:06d}.npz"
                np.savez_compressed(path, region_tokens=result["fused_tokens"][0].cpu().numpy())
                rows.append({"sample_id": item["sample_id"], "case_id": item["case_id"],
                             "path": path.name, "sha256": sha256_file(path),
                             "source_contract": str(dataset.resolve(dataset.rows[index]))})
        metadata["samples"] = rows
    (output / "complete.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": "complete", "output": str(output), "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()
