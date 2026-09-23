#!/usr/bin/env python3
"""Audit Panel1 class availability and compare 128/256px cell context features."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from benchmarks.v5.tools.run_crc_panel1_cell_identity import (
    CLASSES, PREPROCESS, SOURCE, WEIGHTS, label, load_ctranspath,
    score, spatial_split, write_json,
)


def load_manifest(root: Path):
    data = pd.read_csv(root / "cells.csv")
    meta = json.loads((root / "manifest.json").read_text())
    file_hash = hashlib.sha256((root / "cells.csv").read_bytes()).hexdigest()
    if file_hash != meta["manifest_sha256"]:
        raise ValueError("cell manifest checksum mismatch")
    if data.duplicated(["mif_global_x", "mif_global_y"]).any():
        raise ValueError("duplicate selected mIF coordinates")
    train = set(data.loc[data.split.eq("train"), "group"])
    evaluation = set(data.loc[data.split.ne("train"), "group"])
    if train & evaluation:
        raise ValueError("train/eval spatial groups overlap")
    return data, meta


def key(x, y):
    return round(float(x), 3), round(float(y), 3)


def inspect_match_file(item):
    path, split, group = item
    stem = path.name.removesuffix(".csv.gz")
    transform = json.loads((SOURCE / "transforms" / f"{stem}.json").read_text())
    status = str(transform.get("status", "missing_status"))
    result = {"split": split, "group": group, "status": status,
              "raw_counts": [0]*5, "accepted_counts": [0]*5,
              "eligible_counts": [0]*5, "eligible": []}
    if status != "ok":
        return result
    frame = pd.read_csv(path)
    if frame.empty:
        return result
    required = {"accepted_match", "base_cell_type", "he_x", "he_y",
                "he_aligned_x", "he_aligned_y", "mif_x", "mif_y", "x", "y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"incomplete match schema: {path}")
    classes = frame.base_cell_type.fillna("").map(label)
    accepted = frame.accepted_match.astype(str).str.lower().eq("true")
    distance = np.hypot(frame.he_aligned_x-frame.mif_x,
                        frame.he_aligned_y-frame.mif_y)
    eligible = accepted & distance.le(4.) & classes.ge(0)
    eligible &= frame.he_x.between(64, 447) & frame.he_y.between(64, 447)
    result["raw_counts"] = np.bincount(classes[classes.ge(0)].astype(int), minlength=5).tolist()
    result["accepted_counts"] = np.bincount(classes[accepted & classes.ge(0)].astype(int), minlength=5).tolist()
    result["eligible_counts"] = np.bincount(classes[eligible].astype(int), minlength=5).tolist()
    result["eligible"] = [(int(cls), key(x, y)) for cls, x, y in zip(
        classes[eligible], frame.loc[eligible, "x"], frame.loc[eligible, "y"])]
    return result


def audit(a):
    data, meta = load_manifest(a.manifest)
    a.output.mkdir(parents=True, exist_ok=False)
    selected = {(s, c): set() for s in ("train", "val", "test") for c in range(5)}
    for row in data.itertuples():
        selected[(row.split, int(row.label))].add(key(row.mif_global_x, row.mif_global_y))
    source = {(s, c): set() for s in ("train", "val", "test") for c in range(5)}
    lymph_groups = {s: set() for s in ("train", "val", "test")}
    raw = {s: np.zeros(5, np.int64) for s in ("train", "val", "test")}
    accepted = {s: np.zeros(5, np.int64) for s in raw}
    eligible_rows = {s: np.zeros(5, np.int64) for s in raw}
    statuses = {s: Counter() for s in raw}
    candidates, excluded_boundary = [], 0
    for path in (SOURCE / "cell_matching/patch_matches").glob("*.csv.gz"):
        stem = path.name.removesuffix(".csv.gz")
        x, y = map(int, stem.replace("x", "").replace("y", "").split("_"))
        assignment = spatial_split(x, y)
        if assignment is None:
            excluded_boundary += 1
            continue
        candidates.append((path, *assignment))
    candidates.sort(key=lambda item: item[0].name)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for index, result in enumerate(pool.map(inspect_match_file, candidates), 1):
            split = result["split"]
            statuses[split][result["status"]] += 1
            raw[split] += result["raw_counts"]
            accepted[split] += result["accepted_counts"]
            eligible_rows[split] += result["eligible_counts"]
            for cls, coordinate in result["eligible"]:
                source[(split, cls)].add(coordinate)
                if cls == 2:
                    lymph_groups[split].add(result["group"])
            if index % 250 == 0:
                print(json.dumps({"event": "audit_progress", "patches": index,
                    "total": len(candidates), "train_lymphatic_unique": len(source[("train", 2)])}), flush=True)
    available = {}
    for split in raw:
        available[split] = {"source_raw_class_rows": raw[split].tolist(),
            "source_accepted_class_rows": accepted[split].tolist(),
            "source_eligible_class_rows": eligible_rows[split].tolist(),
            "source_eligible_unique_cells": [len(source[(split, c)]) for c in range(5)],
            "selected_unique_cells": [len(selected[(split, c)]) for c in range(5)],
            "eligible_not_selected_unique_cells": [len(source[(split, c)]-selected[(split, c)])
                                                    for c in range(5)],
            "selected_absent_from_eligible_source": [len(selected[(split, c)]-source[(split, c)])
                                                     for c in range(5)],
            "lymphatic_eligible_spatial_groups": len(lymph_groups[split]),
            "patch_transform_status": dict(statuses[split])}
    report = {"status": "complete", "scope": "same-case fixed spatial-split source matches",
        "manifest_sha256": meta["manifest_sha256"], "classes": CLASSES,
        "source_patch_files": len(candidates)+excluded_boundary,
        "spatial_boundary_excluded_patch_files": excluded_boundary,
        "spatial_candidate_patch_files": len(candidates), "by_split": available,
        "interpretation": "raw/accepted/eligible row counts can contain duplicates; unique counts use mIF global coordinates rounded to 0.001px"}
    write_json(a.output / "availability.json", report)
    print(json.dumps({"status": "complete", "lymphatic": {s: {
        "eligible_unique": available[s]["source_eligible_unique_cells"][2],
        "selected": available[s]["selected_unique_cells"][2],
        "not_selected": available[s]["eligible_not_selected_unique_cells"][2]}
        for s in available}}, ensure_ascii=False), flush=True)


def context256(image, mask, row):
    if image.shape != (512, 512, 3) or mask.shape != (512, 512):
        raise ValueError("source H&E or mask must be 512px")
    cx, cy = round(float(row.he_x)), round(float(row.he_y))
    x0, y0 = max(0, min(cx-128, 256)), max(0, min(cy-128, 256))
    view = image[y0:y0+256, x0:x0+256]
    target = mask[y0:y0+256, x0:x0+256] == int(row.he_id)
    if view.shape != (256, 256, 3) or not target.any():
        raise ValueError(f"target nucleus absent from 256px context: {row.patch}/{row.he_id}")
    yy, xx = np.ogrid[:256, :256]
    weight = .65 + .35*np.exp(-((xx-(float(row.he_x)-x0))**2 +
                                (yy-(float(row.he_y)-y0))**2)/(2.*48.**2))
    return (view.astype(np.float32)/255.)*weight[..., None].astype(np.float32)


def extract(a):
    data, meta = load_manifest(a.manifest)
    a.output.mkdir(parents=True, exist_ok=False)
    records = {r["patch"]: r for r in meta["patches"]}
    x = np.full((len(data), 768), np.nan, np.float32)
    torch.set_num_threads(8)
    device = torch.device(a.device)
    encoder = load_ctranspath(WEIGHTS, device).requires_grad_(False)
    chunk_indices, chunk_features, chunks = [], [], []
    processed = 0
    for patch, frame in data.groupby("patch", sort=False):
        record = records[patch]
        image_path = SOURCE / "registered_he_patches" / f"{patch}.png"
        mask_path = SOURCE / "cell_matching/masks" / f"{patch}.npy"
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["he_sha256"]:
            raise ValueError(f"H&E source changed: {patch}")
        if hashlib.sha256(mask_path.read_bytes()).hexdigest() != record["mask_sha256"]:
            raise ValueError(f"nucleus mask source changed: {patch}")
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.load(mask_path, allow_pickle=False)
        tensors = [PREPROCESS(context256(image, mask, row)) for row in frame.itertuples()]
        with torch.inference_mode():
            values = torch.cat([encoder(batch.to(device)).cpu()
                                for batch in torch.stack(tensors).split(32)]).numpy().astype(np.float32)
        if values.shape != (len(frame), 768) or not np.isfinite(values).all():
            raise ValueError(f"invalid 256px features: {patch}")
        x[frame.index] = values
        chunk_indices.extend(frame.index.tolist())
        chunk_features.append(values)
        processed += 1
        if processed % 20 == 0 or (a.max_patches and processed == a.max_patches):
            shard = a.output / f"shard_{len(chunks):04d}.npz"
            np.savez_compressed(shard, indices=np.array(chunk_indices, np.int64),
                                features=np.concatenate(chunk_features))
            chunks.append({"path": shard.name, "patches_through": processed,
                           "cells": len(chunk_indices)})
            chunk_indices, chunk_features = [], []
            print(json.dumps({"event": "context_features", "patches": processed,
                "total_patches": data.patch.nunique(), "cells": int(np.isfinite(x[:, 0]).sum())}), flush=True)
        if a.max_patches and processed >= a.max_patches:
            write_json(a.output / "smoke.json", {"status": "complete", "patches": processed,
                "cells": int(np.isfinite(x[:, 0]).sum()), "feature_dim": 768,
                "manifest_sha256": meta["manifest_sha256"]})
            return
    if chunk_indices:
        shard = a.output / f"shard_{len(chunks):04d}.npz"
        np.savez_compressed(shard, indices=np.array(chunk_indices, np.int64),
                            features=np.concatenate(chunk_features))
        chunks.append({"path": shard.name, "patches_through": processed,
                       "cells": len(chunk_indices)})
    if not np.isfinite(x).all():
        raise ValueError("incomplete 256px feature cache")
    np.savez_compressed(a.output / "features.npz", features=x)
    write_json(a.output / "feature_audit.json", {"status": "complete",
        "manifest_sha256": meta["manifest_sha256"], "cells": len(data), "patches": processed,
        "feature_dim": 768, "context_crop_px": 256,
        "attention": "0.65 + 0.35*Gaussian(sigma=48px) around matched HE nucleus",
        "source_hashes_checked": True, "chunks": chunks})


def compare(a):
    data, meta = load_manifest(a.manifest)
    cached_128 = json.loads((a.features128 / "feature_audit.json").read_text())
    cached_256 = json.loads((a.features256 / "feature_audit.json").read_text())
    if any(report["manifest_sha256"] != meta["manifest_sha256"]
           for report in (cached_128, cached_256)):
        raise ValueError("features do not match fixed cell manifest")
    with np.load(a.features128 / "features.npz", allow_pickle=False) as z:
        raw128 = z["features"]
    with np.load(a.features256 / "features.npz", allow_pickle=False) as z:
        raw256 = z["features"]
    if raw128.shape != raw256.shape or raw128.shape != (len(data), 768):
        raise ValueError("128/256 feature rows are not aligned")
    if not np.isfinite(raw128).all() or not np.isfinite(raw256).all():
        raise ValueError("non-finite features")
    x128 = torch.nn.functional.normalize(torch.from_numpy(raw128), dim=1).numpy()
    x256 = torch.nn.functional.normalize(torch.from_numpy(raw256), dim=1).numpy()
    arms = {"128": x128, "256": x256,
            "128_plus_256": np.concatenate([x128, x256], axis=1)}
    y = data.label.to_numpy(dtype=int)
    ids = {split: np.flatnonzero(data.split.eq(split)) for split in ("train", "val", "test")}
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, features in arms.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1., class_weight="balanced", max_iter=2000))
        model.fit(features[ids["train"]], y[ids["train"]])
        pred_val = model.predict(features[ids["val"]])
        pred_test = model.predict(features[ids["test"]])
        results[name] = {"input_dim": features.shape[1],
            "train_cells": len(ids["train"]),
            "val": score(y[ids["val"]], pred_val),
            "test_exploratory": score(y[ids["test"]], pred_test)}
        if name == "128" and a.baseline_predictions:
            previous = pd.read_csv(a.baseline_predictions)
            actual = data.iloc[ids["test"]]
            expected_ids = list(zip(previous.patch, previous.he_id))
            actual_ids = list(zip(actual.patch, actual.he_id))
            if expected_ids != actual_ids or not np.array_equal(previous.predicted.to_numpy(), pred_test):
                raise ValueError("128px baseline test predictions did not reproduce")
        joblib.dump({"model": model, "classes": CLASSES, "arm": name,
                     "manifest_sha256": meta["manifest_sha256"]}, a.output / f"{name}.joblib")
        table = data.iloc[ids["test"]].copy()
        table["predicted"] = pred_test
        table.to_csv(a.output / f"{name}_test_predictions.csv", index=False)
        print(json.dumps({"arm": name, "val_macro_f1": results[name]["val"]["macro_f1"],
            "test_macro_f1_exploratory": results[name]["test_exploratory"]["macro_f1"]}), flush=True)
    write_json(a.output / "metrics.json", {"status": "complete", "classes": CLASSES,
        "manifest_sha256": meta["manifest_sha256"], "support": meta["support"],
        "primary_selection_split": "val", "test_status": "previously examined; exploratory",
        "classifier": "L2-normalized features, train-only StandardScaler, class-balanced logistic C=1",
        "arms": results})


def compare_weighted(a):
    data, meta = load_manifest(a.manifest)
    audits = [json.loads((root / "feature_audit.json").read_text())
              for root in (a.features128, a.features256)]
    if any(report["manifest_sha256"] != meta["manifest_sha256"] for report in audits):
        raise ValueError("features do not match fixed cell manifest")
    with np.load(a.features128 / "features.npz", allow_pickle=False) as z:
        raw128 = z["features"]
    with np.load(a.features256 / "features.npz", allow_pickle=False) as z:
        raw256 = z["features"]
    if raw128.shape != raw256.shape or raw128.shape != (len(data), 768):
        raise ValueError("feature row mismatch")
    ids = {split: np.flatnonzero(data.split.eq(split)) for split in ("train", "val", "test")}
    y = data.label.to_numpy(dtype=int)
    x128 = torch.nn.functional.normalize(torch.from_numpy(raw128), dim=1).numpy()
    x256 = torch.nn.functional.normalize(torch.from_numpy(raw256), dim=1).numpy()
    scaler128, scaler256 = StandardScaler(), StandardScaler()
    scaler128.fit(x128[ids["train"]]); scaler256.fit(x256[ids["train"]])
    z128, z256 = scaler128.transform(x128), scaler256.transform(x256)
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for weight in (0.25, 0.5, 0.75, 1.0):
        features = np.concatenate([z128, weight*z256], axis=1)
        model = LogisticRegression(C=1., class_weight="balanced", max_iter=2000)
        model.fit(features[ids["train"]], y[ids["train"]])
        val = score(y[ids["val"]], model.predict(features[ids["val"]]))
        test = score(y[ids["test"]], model.predict(features[ids["test"]]))
        name = f"context_weight_{weight:.2f}"
        results[name] = {"weight": weight, "val": val, "test_exploratory": test}
        joblib.dump({"model": model, "scaler128": scaler128,
                     "scaler256": scaler256, "context_weight": weight,
                     "manifest_sha256": meta["manifest_sha256"]}, a.output / f"{name}.joblib")
        print(json.dumps({"arm": name, "val_macro_f1": val["macro_f1"],
                          "test_macro_f1_exploratory": test["macro_f1"]}), flush=True)
    selected = max(results, key=lambda name: results[name]["val"]["macro_f1"])
    write_json(a.output / "metrics.json", {"status": "complete", "classes": CLASSES,
        "manifest_sha256": meta["manifest_sha256"], "weights": [0.25, 0.5, 0.75, 1.0],
        "scalers_fit_on": "train only", "selected_context_weight_on_val": selected,
        "test_status": "previously examined; exploratory", "arms": results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("audit", "extract", "compare", "compare_weighted"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--max-patches", type=int, default=0)
    p.add_argument("--features128", type=Path)
    p.add_argument("--features256", type=Path)
    p.add_argument("--baseline-predictions", type=Path)
    a = p.parse_args()
    if not a.output.resolve().is_relative_to("/nfs-medical3/zyh/fewclick_v2"):
        p.error("output must be a timestamped FewClick artifact directory")
    if a.mode in ("compare", "compare_weighted") and (a.features128 is None or a.features256 is None):
        p.error("comparison requires --features128 and --features256")
    {"audit": audit, "extract": extract, "compare": compare,
     "compare_weighted": compare_weighted}[a.mode](a)


if __name__ == "__main__":
    main()
