#!/usr/bin/env python3
"""Create a label-stratified H&E context audit for the CRC Panel1 cell probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, SOURCE

SCALES = (128, 256, 512)
TILE = 256


def order_key(row, salt: str) -> str:
    key = f"{salt}/{row.patch}/{int(row.he_id)}"
    return hashlib.sha256(key.encode()).hexdigest()


def choose(frame: pd.DataFrame, salt: str) -> pd.Series | None:
    if frame.empty:
        return None
    ranked = frame.copy()
    ranked["_hash"] = [order_key(r, salt) for r in ranked.itertuples()]
    ranked["_distance_rank"] = (ranked.distance > 2.0).astype(int)
    return ranked.sort_values(["_distance_rank", "_hash"]).iloc[0].drop(labels=["_hash", "_distance_rank"])


def font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def crop(image: np.ndarray, cx: int, cy: int, size: int, pad_value=0) -> np.ndarray:
    h, w = image.shape[:2]
    half = size // 2
    x0, y0 = cx-half, cy-half
    x1, y1 = x0+size, y0+size
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1-w), max(0, y1-h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    view = image[y0:y1, x0:x1]
    if any((pad_l, pad_t, pad_r, pad_b)):
        view = np.pad(view, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                      mode="constant", constant_values=pad_value)
    if view.shape[:2] != (size, size):
        raise ValueError(f"bad {size}px crop geometry: {view.shape}")
    return view


def marked_crop(image, mask, row, size):
    cx, cy = int(round(float(row.he_x))), int(round(float(row.he_y)))
    if size == 512:
        # The source is one registered 512px tile: show the real tile instead
        # of a centered crop padded with repeated pixels at its borders.
        rgb = image.copy()
        target = mask == int(row.he_id)
    else:
        rgb = crop(image, cx, cy, size, pad_value=255)
        target = crop(mask[..., None], cx, cy, size, pad_value=0)[..., 0] == int(row.he_id)
    if not target.any():
        raise ValueError(f"target nucleus absent from context crop: {row.patch}/{row.he_id}/{size}")
    edge = target & ~ndimage.binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
    rgb = rgb.copy()
    rgb[edge] = (0, 255, 255)
    return Image.fromarray(rgb).resize((TILE, TILE), Image.Resampling.LANCZOS)


def make_detail(class_name, samples, output):
    label_font, small_font = font(20), font(15)
    margin, title_h, sample_h, tile_h = 16, 38, 26, TILE + 25
    width = margin * 2 + 3 * (TILE + 8)
    height = margin + title_h + len(samples) * (sample_h + tile_h + 8) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), class_name, fill="black", font=label_font)
    y = margin + title_h
    for sample in samples:
        split, row = sample["split"], sample["row"]
        pred = "—" if split == "train-added" else CLASSES[int(row.predicted)]
        text = (f"{split}; true={class_name}; pred={pred}; "
                f"patch={row.patch}; HE id={int(row.he_id)}; match={float(row.distance):.2f}px")
        draw.text((margin, y), text, fill="#111111", font=small_font)
        y += sample_h
        image_path = SOURCE / "registered_he_patches" / f"{row.patch}.png"
        mask_path = SOURCE / "cell_matching/masks" / f"{row.patch}.npy"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.load(mask_path, allow_pickle=False)
        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            raise ValueError(f"unexpected source geometry: {row.patch}")
        target = mask == int(row.he_id)
        if not target.any():
            raise ValueError(f"HE nucleus id missing: {row.patch}/{row.he_id}")
        nuc_y, nuc_x = ndimage.center_of_mass(target)
        centroid_error = float(np.hypot(nuc_x-row.he_x, nuc_y-row.he_y))
        if centroid_error > 0.1:
            raise ValueError(f"HE centroid mismatch: {row.patch}/{row.he_id}/{centroid_error}")
        for col, scale in enumerate(SCALES):
            x = margin + col * (TILE + 8)
            caption = "512px full tile" if scale == 512 else f"{scale} x {scale} px, centered"
            draw.text((x, y), caption, fill="#222222", font=small_font)
            canvas.paste(marked_crop(image, mask, row, scale), (x, y + 20))
        y += tile_h + 8
        yield {"class": class_name, "split": split, "patch": row.patch,
               "group": row.group, "he_id": int(row.he_id), "mif_id": int(row.mif_id),
               "source_label": row.source_label, "true_label": class_name,
               "predicted_label": pred, "distance_px": float(row.distance),
               "he_centroid_error_px": centroid_error, "he_x": float(row.he_x),
               "he_y": float(row.he_y), "128_path": str(image_path),
               "mask_path": str(mask_path)}
    canvas.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--added-patch-audit", type=Path, required=True)
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    allowed = Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    if not a.output.resolve().is_relative_to(allowed):
        p.error("output must be under /nfs-medical3/zyh/fewclick_v2")
    a.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(a.manifest / "cells.csv")
    added = pd.read_csv(a.added_patch_audit)
    test = pd.read_csv(a.test_predictions)
    added_names = set(added.patch)
    train = data[data.split.eq("train") & data.patch.isin(added_names)]
    samples_by_class, audit_rows = {}, []
    for class_id, class_name in enumerate(CLASSES):
        chosen = []
        rows = train[train.label.eq(class_id)]
        for n in range(2):
            row = choose(rows, f"context-train-{class_id}-{n}-20260923")
            if row is not None:
                chosen.append({"split": "train-added", "row": row})
                rows = rows.drop(index=row.name)
        rows = test[test.label.eq(class_id)].copy()
        correct = rows[rows.predicted.eq(class_id)]
        wrong = rows[~rows.predicted.eq(class_id)]
        for tag, pool in (("test-correct", correct), ("test-error", wrong)):
            row = choose(pool, f"context-{tag}-{class_id}-20260923")
            if row is not None:
                chosen.append({"split": tag, "row": row})
        # When the entire test class is too small to include both outcomes,
        # show every available cell in addition to train examples.
        if len(rows) <= 2:
            for _, row in rows.iterrows():
                if not any(s["split"].startswith("test") and s["row"].patch == row.patch
                           and int(s["row"].he_id) == int(row.he_id) for s in chosen):
                    chosen.append({"split": "test-error", "row": row})
        if len(chosen) < 2:
            raise ValueError(f"too few visual samples for {class_name}")
        samples_by_class[class_name] = chosen
        audit_rows.extend(make_detail(class_name, chosen, a.output / f"{class_name}.png"))

    # One test example per class for a compact overview; use a model error
    # when possible so the montage does not show only easy cases.
    overview_w, row_h, margin = 3 * (TILE + 8) + 24, TILE + 64, 14
    overview = Image.new("RGB", (overview_w, margin + len(CLASSES)*row_h), "white")
    draw = ImageDraw.Draw(overview)
    title_font, label_font = font(19), font(15)
    for class_id, class_name in enumerate(CLASSES):
        y = margin + class_id * row_h
        draw.text((margin, y), class_name, fill="black", font=title_font)
        row_samples = samples_by_class[class_name]
        sample = next((s for s in row_samples if s["split"] == "test-error"),
                      next((s for s in row_samples if s["split"].startswith("test")), row_samples[0]))
        row = sample["row"]
        image = np.asarray(Image.open(SOURCE / "registered_he_patches" / f"{row.patch}.png").convert("RGB"))
        mask = np.load(SOURCE / "cell_matching/masks" / f"{row.patch}.npy", allow_pickle=False)
        for col, scale in enumerate(SCALES):
            x = margin + col * (TILE + 8)
            caption = "512px full tile" if scale == 512 else f"{scale}px centered"
            draw.text((x, y + 25), caption, fill="#222222", font=label_font)
            overview.paste(marked_crop(image, mask, row, scale), (x, y + 45))
    overview.save(a.output / "all_classes_context_overview.png")
    pd.DataFrame(audit_rows).to_csv(a.output / "sample_audit.csv", index=False)
    metadata = {"status": "complete", "classes": CLASSES,
        "scales_px": SCALES, "model_crop_px": 128,
        "source": str(SOURCE), "samples": len(audit_rows),
        "sampling": "up to 2 new train cells and up to 1 correct/1 incorrect fixed-test example per class; both test cells retained when support <=2",
        "note": "cyan contour marks the registered H&E nucleus mask; labels are mIF-derived matched-cell labels"}
    (a.output / "audit.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
