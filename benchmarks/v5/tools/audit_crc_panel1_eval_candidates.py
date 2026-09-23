#!/usr/bin/env python3
"""Verify unused Panel1 matched cells and build a spatially held-out evaluation supplement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, SOURCE, label, spatial_split, write_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(x, y):
    return round(float(x), 3), round(float(y), 3)


def scan(manifest: Path, output: Path, per_class: int):
    original = pd.read_csv(manifest / "cells.csv")
    meta = json.loads((manifest / "manifest.json").read_text())
    if sha(manifest / "cells.csv") != meta["manifest_sha256"]:
        raise ValueError("manifest checksum mismatch")
    old_keys = {key(r.mif_global_x, r.mif_global_y) for r in original.itertuples()}
    old_patches = set(original.patch)
    train_groups = set(original.loc[original.split.eq("train"), "group"])
    eval_groups = set(original.loc[original.split.ne("train"), "group"])
    if train_groups & eval_groups:
        raise ValueError("train/eval group collision")
    candidates = defaultdict(list)
    lymph_audit = []
    counters = defaultdict(Counter)
    paths = sorted((SOURCE / "cell_matching/patch_matches").glob("*.csv.gz"))
    for index, path in enumerate(paths, 1):
        stem = path.name.removesuffix(".csv.gz")
        x, y = map(int, stem.replace("x", "").replace("y", "").split("_"))
        assignment = spatial_split(x, y)
        if assignment is None or assignment[0] not in ("val", "test"):
            continue
        split, group = assignment
        transform = json.loads((SOURCE / "transforms" / f"{stem}.json").read_text())
        if transform.get("status") != "ok":
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        required = {"accepted_match", "base_cell_type", "he_cell_id", "he_x", "he_y",
                    "he_aligned_x", "he_aligned_y", "mif_x", "mif_y", "x", "y", "cell_id"}
        if not required.issubset(frame):
            raise ValueError(f"incomplete match schema: {path}")
        classes = frame.base_cell_type.fillna("").map(label)
        distance = np.hypot(frame.he_aligned_x - frame.mif_x, frame.he_aligned_y - frame.mif_y)
        accepted = frame.accepted_match.astype(str).str.lower().eq("true")
        eligible = accepted & distance.le(4) & classes.ge(0)
        eligible &= frame.he_x.between(64, 447) & frame.he_y.between(64, 447)
        eligible_rows = frame.loc[eligible].copy()
        if eligible_rows.empty:
            continue
        eligible_rows["label"] = classes[eligible].astype(int)
        eligible_rows["distance"] = distance[eligible]
        # Audit every unused lymphatic candidate. Queue other classes cheaply;
        # only sampled rows need their source mask loaded and checked.
        if stem not in old_patches:
            for cell in eligible_rows.loc[eligible_rows.label.ne(2)].itertuples():
                if key(cell.x, cell.y) in old_keys:
                    continue
                cls = int(cell.label)
                counters[split][f"candidate_{cls}"] += 1
                candidates[(split, cls)].append(dict(patch=stem, split=split, group=group,
                    he_id=int(cell.he_cell_id), mif_id=int(cell.cell_id), he_x=float(cell.he_x),
                    he_y=float(cell.he_y), mif_global_x=float(cell.x), mif_global_y=float(cell.y),
                    label=cls, source_label=str(cell.base_cell_type), distance=float(cell.distance),
                    old_patch=False, mask_status="pending"))
        check_rows = eligible_rows.loc[eligible_rows.label.eq(2)]
        if check_rows.empty:
            continue
        image_path = SOURCE / "registered_he_patches" / f"{stem}.png"
        mask_path = SOURCE / "cell_matching/masks" / f"{stem}.npy"
        if not image_path.is_file() or not mask_path.is_file():
            counters[split]["missing_source_patch"] += 1
            continue
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != (512, 512) or Image.open(image_path).size != (512, 512):
            raise ValueError(f"bad source geometry: {stem}")
        ids = {int(v): np.flatnonzero(mask == int(v)) for v in check_rows.he_cell_id.unique()}
        hashes = None
        for cell in check_rows.itertuples():
            coordinate = key(cell.x, cell.y)
            if coordinate in old_keys:
                continue
            cls = int(cell.label)
            counters[split][f"candidate_{cls}"] += 1
            reason = "ok"
            positions = ids[int(cell.he_cell_id)]
            if not len(positions):
                reason = "missing_nucleus"
            else:
                yy, xx = np.divmod(positions, 512)
                if np.hypot(xx.mean() - cell.he_x, yy.mean() - cell.he_y) > .1:
                    reason = "centroid_mismatch"
            record = dict(patch=stem, split=split, group=group,
                he_id=int(cell.he_cell_id), mif_id=int(cell.cell_id), he_x=float(cell.he_x),
                he_y=float(cell.he_y), mif_global_x=float(cell.x), mif_global_y=float(cell.y),
                label=cls, source_label=str(cell.base_cell_type), distance=float(cell.distance),
                old_patch=stem in old_patches, mask_status=reason)
            if cls == 2:
                lymph_audit.append(record)
            if reason != "ok":
                counters[split][reason] += 1
                continue
            counters[split][f"verified_{cls}"] += 1
            if stem in old_patches:
                continue
            if hashes is None:
                hashes = dict(matches_sha256=sha(path), he_sha256=sha(image_path), mask_sha256=sha(mask_path))
            record.update(hashes)
            candidates[(split, cls)].append(record)
        if index % 1000 == 0:
            print(json.dumps({"event": "scan", "source_files_seen": index}), flush=True)

    # One mIF identity may appear in overlapping registered patches. Retain one
    # representation globally and spread selected cells over available groups.
    selected, dedup = [], set(old_keys)
    checked_patches = {}
    availability = {}
    for split in ("val", "test"):
        for cls in range(5):
            rows = sorted(candidates[(split, cls)], key=lambda r: (r["group"], r["patch"], r["mif_global_x"], r["mif_global_y"]))
            buckets = defaultdict(list)
            for row in rows:
                buckets[row["group"]].append(row)
            groups = sorted(buckets)
            availability[f"{split}_{CLASSES[cls]}"] = {"raw_verified_rows": len(rows), "groups": len(groups)}
            count = 0
            while count < per_class and groups:
                next_groups = []
                for group in groups:
                    while buckets[group]:
                        row = buckets[group].pop(0)
                        coordinate = key(row["mif_global_x"], row["mif_global_y"])
                        if coordinate not in dedup:
                            if row["mask_status"] == "pending":
                                patch = row["patch"]
                                if patch not in checked_patches:
                                    image_path = SOURCE / "registered_he_patches" / f"{patch}.png"
                                    mask_path = SOURCE / "cell_matching/masks" / f"{patch}.npy"
                                    matches_path = SOURCE / "cell_matching/patch_matches" / f"{patch}.csv.gz"
                                    if not image_path.is_file() or not mask_path.is_file():
                                        checked_patches[patch] = None
                                    else:
                                        mask = np.load(mask_path, allow_pickle=False)
                                        if mask.shape != (512, 512) or Image.open(image_path).size != (512, 512):
                                            raise ValueError(f"bad source geometry: {patch}")
                                        checked_patches[patch] = (mask, dict(
                                            matches_sha256=sha(matches_path), he_sha256=sha(image_path),
                                            mask_sha256=sha(mask_path)))
                                source = checked_patches[patch]
                                if source is None:
                                    counters[split]["missing_source_patch"] += 1
                                    continue
                                mask, hashes = source
                                positions = np.flatnonzero(mask == row["he_id"])
                                if not len(positions):
                                    counters[split]["missing_nucleus"] += 1
                                    continue
                                yy, xx = np.divmod(positions, 512)
                                if np.hypot(xx.mean()-row["he_x"], yy.mean()-row["he_y"]) > .1:
                                    counters[split]["centroid_mismatch"] += 1
                                    continue
                                row.update(hashes)
                                row["mask_status"] = "ok"
                                counters[split][f"verified_{cls}"] += 1
                            selected.append(row)
                            dedup.add(coordinate)
                            count += 1
                            break
                    if buckets[group]:
                        next_groups.append(group)
                    if count >= per_class:
                        break
                groups = next_groups
            availability[f"{split}_{CLASSES[cls]}"]["selected"] = count
    output.mkdir(parents=True, exist_ok=False)
    columns = ["patch", "split", "group", "he_id", "mif_id", "he_x", "he_y",
               "mif_global_x", "mif_global_y", "label", "source_label", "distance",
               "old_patch", "mask_status", "matches_sha256", "he_sha256", "mask_sha256"]
    pd.DataFrame(selected, columns=columns).to_csv(output / "cells.csv", index=False)
    pd.DataFrame(lymph_audit).to_csv(output / "lymphatic_candidate_audit.csv", index=False)
    result = {"status": "complete", "source_manifest_sha256": meta["manifest_sha256"],
              "manifest_sha256": sha(output / "cells.csv"), "old_selected_excluded": True,
              "old_eval_patches_excluded_from_supplement": True,
              "same_case_spatial_split": True, "per_class_cap": per_class,
              "support": {s: np.bincount([r["label"] for r in selected if r["split"] == s], minlength=5).tolist()
                          for s in ("val", "test")},
              "verified_candidate_rows": {s: dict(counters[s]) for s in ("val", "test")},
              "availability": availability,
              "selected_groups": {s: len({r["group"] for r in selected if r["split"] == s})
                                  for s in ("val", "test")}}
    write_json(output / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=50)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to("/nfs-medical3/zyh/fewclick_v2"):
        parser.error("output must be within FewClick artifact root")
    scan(args.manifest, args.output, args.per_class)


if __name__ == "__main__":
    main()
