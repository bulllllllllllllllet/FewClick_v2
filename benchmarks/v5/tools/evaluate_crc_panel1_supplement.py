#!/usr/bin/env python3
"""Encode independently sampled Panel1 evaluation cells and score frozen heads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import crop_mask, crop_region, soft_attention
from benchmarks.v5.tools.probe_crc_panel1_multiscale_context import context256
from benchmarks.v5.tools.run_crc_panel1_cell_identity import (
    CLASSES, PREPROCESS, SOURCE, WEIGHTS, load_ctranspath, score, write_json,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-manifest", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--weighted-models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    if not a.output.resolve().is_relative_to("/nfs-medical3/zyh/fewclick_v2"):
        p.error("timestamped FewClick artifact output required")
    data = pd.read_csv(a.manifest / "cells.csv")
    meta = json.loads((a.manifest / "audit.json").read_text())
    original = json.loads((a.original_manifest / "manifest.json").read_text())
    if sha(a.manifest / "cells.csv") != meta["manifest_sha256"]:
        raise ValueError("supplement manifest checksum mismatch")
    if meta["source_manifest_sha256"] != original["manifest_sha256"]:
        raise ValueError("source manifest changed")
    if data.empty or data.old_patch.any() or not data.mask_status.eq("ok").all():
        raise ValueError("supplement not isolated or mask audit failed")
    old_data = pd.read_csv(a.original_manifest / "cells.csv")
    old = set(zip(old_data.mif_global_x.round(3), old_data.mif_global_y.round(3)))
    new = set(zip(data.mif_global_x.round(3), data.mif_global_y.round(3)))
    if old & new or len(new) != len(data):
        raise ValueError("cell identity overlap or duplicate")
    train_groups = set(old_data.loc[old_data.split.eq("train"), "group"])
    if train_groups & set(data.group):
        raise ValueError("train/eval group overlap")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    device = torch.device(a.device)
    encoder = load_ctranspath(WEIGHTS, device).requires_grad_(False)
    features = {128: np.full((len(data), 768), np.nan, np.float32),
                256: np.full((len(data), 768), np.nan, np.float32)}
    for index, (patch, frame) in enumerate(data.groupby("patch", sort=False), 1):
        row = frame.iloc[0]
        paths = {"matches_sha256": SOURCE / "cell_matching/patch_matches" / f"{patch}.csv.gz",
                 "he_sha256": SOURCE / "registered_he_patches" / f"{patch}.png",
                 "mask_sha256": SOURCE / "cell_matching/masks" / f"{patch}.npy"}
        if any(sha(path) != row[name] for name, path in paths.items()):
            raise ValueError(f"source changed: {patch}")
        image = np.asarray(Image.open(paths["he_sha256"]).convert("RGB"))
        mask = np.load(paths["mask_sha256"], allow_pickle=False)
        for size in (128, 256):
            if size == 128:
                tensors = [PREPROCESS(soft_attention(
                    crop_region(image, round(cell.he_y), round(cell.he_x)),
                    crop_mask(mask, round(cell.he_y), round(cell.he_x)) == int(cell.he_id)))
                    for cell in frame.itertuples()]
            else:
                tensors = [PREPROCESS(context256(image, mask, cell)) for cell in frame.itertuples()]
            with torch.inference_mode():
                values = torch.cat([encoder(batch.to(device)).cpu()
                                    for batch in torch.stack(tensors).split(32)]).numpy().astype(np.float32)
            features[size][frame.index] = values
        if index % 20 == 0:
            print(json.dumps({"event": "encoded", "patches": index,
                              "total": data.patch.nunique()}), flush=True)
    if any(not np.isfinite(x).all() for x in features.values()):
        raise ValueError("incomplete feature extraction")
    np.savez_compressed(a.output / "features.npz", features128=features[128], features256=features[256])
    norm = {size: torch.nn.functional.normalize(torch.from_numpy(x), dim=1).numpy()
            for size, x in features.items()}
    y = data.label.to_numpy(dtype=int)
    arms = {"128": norm[128], "256": norm[256],
            "128_plus_256": np.concatenate([norm[128], norm[256]], axis=1)}
    results = {}
    for name, x in arms.items():
        packed = joblib.load(a.models / f"{name}.joblib")
        if packed["manifest_sha256"] != original["manifest_sha256"]:
            raise ValueError("frozen model manifest mismatch")
        pred = packed["model"].predict(x)
        results[name] = {s: score(y[data.split.eq(s)], pred[data.split.eq(s)])
                         for s in ("val", "test")}
        table = data.copy(); table["predicted"] = pred
        table.to_csv(a.output / f"{name}_predictions.csv", index=False)
    packed = joblib.load(a.weighted_models / "context_weight_0.25.joblib")
    if packed["manifest_sha256"] != original["manifest_sha256"]:
        raise ValueError("weighted model manifest mismatch")
    x = np.concatenate([packed["scaler128"].transform(norm[128]),
                        .25 * packed["scaler256"].transform(norm[256])], axis=1)
    pred = packed["model"].predict(x)
    results["128_plus_0.25x256"] = {s: score(y[data.split.eq(s)], pred[data.split.eq(s)])
                                    for s in ("val", "test")}
    table = data.copy(); table["predicted"] = pred
    table.to_csv(a.output / "128_plus_0.25x256_predictions.csv", index=False)
    write_json(a.output / "metrics.json", {"status": "complete", "classes": CLASSES,
        "manifest_sha256": meta["manifest_sha256"], "source_model_manifest_sha256": original["manifest_sha256"],
        "evaluation_scope": "new cells and unused patches in same-case fixed val/test spatial groups; exploratory",
        "support": meta["support"], "arms": results})
    print(json.dumps({name: {s: {"macro_f1": r[s]["macro_f1"], "accuracy": r[s]["accuracy"]}
                             for s in r} for name, r in results.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
