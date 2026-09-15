#!/usr/bin/env python3
"""Audit complete CRC candidate caches; never promote weak labels to truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(root):
    meta = json.loads((root / "metadata.json").read_text())
    classes = meta["identity_classes"]
    totals, seen = {}, set()
    grouped_patches = 0
    for record in meta["records"]:
        path = root / record["path"]
        if path.resolve().parent != root.resolve() or path.name in seen:
            raise ValueError("invalid or duplicate cache path")
        seen.add(path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            n = record["he_cells"]
            features, ids = data["reg_features"], data["he_cell_ids"]
            keep, labels = data["candidate_valid"], data["candidate_class"]
            assert features.shape == (n, 64) and np.isfinite(features).all()
            assert ids.shape == (n,) and len(np.unique(ids)) == n
            np.testing.assert_array_equal(np.sort(ids), np.unique(data["he_mask"])[np.unique(data["he_mask"]) > 0])
            assert data["he_xy"].shape == (n, 2) and np.isfinite(data["he_xy"]).all()
            assert keep.shape == labels.shape == (n,) and keep.dtype == np.bool_
            assert data["identity_verified"].shape == (n,) and not data["identity_verified"].any()
            assert record["verified_training_sample"] is False
            assert np.all((labels[keep] >= 0) & (labels[keep] < len(classes)))
            assert np.isfinite(data["match_distance_px"]).all()
            assert np.all(data["match_distance_px"][keep] <= 4)
            assert len(np.unique(data["matched_dapi_id"][keep])) == keep.sum()
            np.testing.assert_array_equal(data["xcell_group_id"], np.arange(n) // 255)
            counts = np.bincount(labels[keep], minlength=len(classes))
            np.testing.assert_array_equal(counts, record["candidate_class_counts"])
            assert int(keep.sum()) == record["candidate_labels"]
            grouped_patches += int(n > 255)
            row = totals.setdefault(record["case_id"], {"patches": 0, "he_cells": 0, "candidates": 0, **dict.fromkeys(classes, 0)})
            row["patches"] += 1
            row["he_cells"] += n
            row["candidates"] += int(keep.sum())
            for name, count in zip(classes, counts):
                row[name] += int(count)
    expected = meta["config"]["per_case"]
    complete = all(totals.get(case, {}).get("patches", 0) == expected for case in meta["config"]["cases"])
    assert (meta["status"] == "complete") == complete
    return {"status": "passed", "extraction_status": meta["status"], "patches": len(seen),
            "grouped_patches": grouped_patches, "by_case": totals,
            "scope": "technical integrity only; no biological identity or registration acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.probe_dir.resolve()
    if not root.is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use FewClick artifact root")
    result = validate(root)
    with (root / "integrity_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    with (root / "class_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", *next(iter(result["by_case"].values())).keys()])
        writer.writeheader()
        for case, row in result["by_case"].items():
            writer.writerow({"case_id": case, **row})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
