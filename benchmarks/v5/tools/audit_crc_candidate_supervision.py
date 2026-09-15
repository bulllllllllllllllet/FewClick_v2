#!/usr/bin/env python3
"""Read-only source audit of CRC weak labels, geometry and raw mIF intensities."""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import expand_labels
import tifffile
import zarr

MARKERS = ["CD4", "CD8", "CD20", "CD163", "FOXP3", "SULF1"]
RANGES = np.array([[300, 2000], [100, 1200], [150, 1200], [232, 1264], [1500, 3500], [500, 3000]])
CLASSES = ["CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage"]
CHANNELS = {"DAPI": "DAPI", "Opal 480": "CD163", "Opal 520": "FOXP3", "Opal 570": "CD20",
            "Opal 620": "CD8", "Opal 690": "CD4", "Opal 780": "SULF1", "Sample AF": "AF"}


def classify(values, factor=1.0, lower_only=False):
    positive = values >= RANGES[:, 0] * factor
    if not lower_only:
        positive &= values <= RANGES[:, 1] * factor
    winner = np.argmax(np.where(positive[:, :4], values[:, :4], -np.inf), axis=1)
    labels = np.full(len(values), -1, dtype=int)
    has = positive[:, :4].any(axis=1)
    labels[has] = np.array([0, 1, 3, 4])[winner[has]]
    labels[has & (winner == 0) & positive[:, 4]] = 2
    base = np.array([CLASSES[i] if i >= 0 else "Undefined" for i in labels], dtype=object)
    base[~has & positive[:, 4]] = "FOXP3_only_Unresolved"
    full = base.copy()
    for i in np.flatnonzero(positive[:, 5]):
        full[i] = "SULF1_only" if base[i] == "Undefined" else base[i] + "_SULF1"
    return labels, full, positive, winner


def geometry(he, dapi, ids, xy, target_ids):
    dids = np.unique(dapi)[1:] if dapi.min() == 0 else np.unique(dapi)
    dyx = np.asarray(ndi.center_of_mass(np.ones(dapi.shape), dapi, dids))
    dxy = dyx[:, ::-1]
    distance, index = cKDTree(dxy).query(xy, k=2)
    _, reverse = cKDTree(xy).query(dxy)
    mutual = reverse[index[:, 0]] == np.arange(len(xy))
    np.testing.assert_array_equal(dids[index[:, 0]], target_ids)
    ha = np.bincount(he.ravel())
    da = np.bincount(dapi.ravel())
    radix = int(dapi.max()) + 1
    pairs, count = np.unique(he.astype(np.int64) * radix + dapi, return_counts=True)
    overlap = dict(zip(pairs.tolist(), count.tolist()))
    inter = np.array([overlap.get(int(h) * radix + int(d), 0) for h, d in zip(ids, target_ids)])
    iou = inter / (ha[ids] + da[target_ids] - inter)
    ratio = ha[ids] / da[target_ids]
    return distance, mutual, iou, ratio, dxy, dids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp required")
    out = Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3") / f"crc_candidate_qc_{args.timestamp}"
    out.mkdir(exist_ok=False)
    meta = json.loads((args.probe_dir / "metadata.json").read_text())
    root = Path(meta["registration_root"])
    all_cells, patches, raw_checks, review = [], [], [], []
    for case in meta["config"]["cases"]:
        qpath = Path("/nfs-medical6/data/SRRSH/crc/panel_2/mIF") / case / "Scan1" / f"{case}_Scan1.er.qptiff"
        with tifffile.TiffFile(qpath) as tif:
            names = []
            for page in tif.pages:
                try:
                    xml = ET.fromstring(page.description or "")
                except ET.ParseError:
                    continue
                if xml.findtext("ImageType") == "FullResolution":
                    names.append(xml.findtext("Name"))
            mapping = {CHANNELS[name]: i for i, name in enumerate(names)}
            assert len(mapping) == 8
            store = tif.series[0].aszarr()
            image = zarr.open(store, mode="r")["0"]
            for record in [r for r in meta["records"] if r["case_id"] == case]:
                stem = record["patch_name"]
                folder = "cell_annotations_mif_only" if case == "2019-00538F" else "cell_annotations"
                table_path = root / case / folder / "cell_tables" / f"{stem}_cells.tsv"
                frame = pd.read_csv(table_path, sep="\t").set_index("cell_id").sort_index()
                transform = json.loads(Path(record["transform_path"]).read_text())
                dapi = np.load(root / case / "dapi_masks" / f"{stem}.npy")
                dids = np.unique(dapi); dids = dids[dids > 0]
                np.testing.assert_array_equal(frame.index, dids)
                values = frame[[f"{m}_intensity" for m in MARKERS]].to_numpy()
                labels, full, positive, winner = classify(values)
                label_mismatch = int((full != frame.cell_type.to_numpy()).sum())
                positive_mismatch = int((positive != frame[[f"{m}_positive" for m in MARKERS]].to_numpy()).sum())
                stored_means = np.maximum(frame[[f"{m}_nuclear_mean" for m in MARKERS]].to_numpy(),
                                          frame[[f"{m}_ring_mean" for m in MARKERS]].to_numpy())
                assert np.allclose(values, stored_means, atol=1e-3)
                x, y = int(transform["x"]), int(transform["y"])
                stack = np.asarray(image[:, y:y+dapi.shape[0], x:x+dapi.shape[1]])
                ring = np.where(dapi == 0, expand_labels(dapi, distance=4), 0)
                max_error = 0.0
                for marker in ["DAPI", *MARKERS, "AF"]:
                    for suffix, mask in [("nuclear_mean", dapi), ("ring_mean", ring)]:
                        measured = np.nan_to_num(ndi.mean(stack[mapping[marker]].astype(np.float32), mask, dids)).astype(np.float32)
                        observed = frame[f"{marker}_{suffix}"].to_numpy()
                        max_error = max(max_error, float(np.max(np.abs(measured-observed))))
                        np.testing.assert_allclose(measured, observed, rtol=2e-6, atol=2e-3)
                raw_checks.append({"case": case, "patch": stem, "qptiff": str(qpath), "channels": mapping,
                                   "max_serialization_error": max_error,
                                   "source_table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()})
                with np.load(args.probe_dir / record["path"]) as data:
                    keep, targets = data["candidate_valid"], data["matched_dapi_id"]
                    distance, mutual, iou, ratio, dxy, dapi_ids = geometry(data["he_mask"], dapi, data["he_cell_ids"], data["he_xy"], targets)
                    target_indices = frame.index.get_indexer(targets)
                    np.testing.assert_array_equal(data["candidate_class"], labels[target_indices])
                    np.testing.assert_allclose(data["match_distance_px"], distance[:, 0], atol=1e-9)
                    np.testing.assert_array_equal(keep, mutual & (distance[:, 0] <= 4) & (labels[target_indices] >= 0))
                    chosen = target_indices[keep]
                    candidates = frame.iloc[chosen]
                    xy = data["he_xy"][keep]
                    cells = pd.DataFrame({"case": case, "patch": stem, "he_id": data["he_cell_ids"][keep],
                                          "dapi_id": targets[keep], "class": np.array(CLASSES)[labels[chosen]],
                                          "x": xy[:, 0], "y": xy[:, 1], "distance": distance[keep, 0],
                                          "second_distance": distance[keep, 1], "iou": iou[keep], "he_dapi_area_ratio": ratio[keep],
                                          "competing_markers": positive[chosen, :4].sum(axis=1),
                                          "changed_minus10pct": classify(values, .9)[0][chosen] != labels[chosen],
                                          "changed_plus10pct": classify(values, 1.1)[0][chosen] != labels[chosen],
                                          "changed_lower_only": classify(values, lower_only=True)[0][chosen] != labels[chosen],
                                          "foxp3_nuclear_positive": candidates.FOXP3_nuclear_mean.between(1500, 3500).to_numpy(),
                                          "foxp3_ring_dominates": (candidates.FOXP3_ring_mean > candidates.FOXP3_nuclear_mean).to_numpy()})
                    cells["geometry_review_pass"] = (cells.distance <= 2) & (cells.second_distance >= 2*cells.distance) & (cells.iou >= .5) & cells.he_dapi_area_ratio.between(.5, 2)
                    cells["stable_single_marker"] = (cells.competing_markers == 1) & ~cells.changed_minus10pct & ~cells.changed_plus10pct
                    for marker in MARKERS:
                        cells[f"{marker}_intensity"] = candidates[f"{marker}_intensity"].to_numpy()
                    all_cells.append(cells)
                    # Null control is spatial displacement, not an estimate of false discovery rate.
                    null = []
                    for shift in [(32, 0), (-32, 0), (0, 32), (0, -32)]:
                        shifted = data["he_xy"] + shift
                        inside = np.all((shifted >= 0) & (shifted < np.array(dapi.shape[::-1])), axis=1)
                        query = shifted[inside]
                        dd, ii = cKDTree(dxy).query(query)
                        _, rr = cKDTree(query).query(dxy)
                        null.append(float(np.mean((dd <= 4) & (rr[ii] == np.arange(len(query))) & (labels[frame.index.get_indexer(dapi_ids[ii])] >= 0))))
                    patches.append({"case": case, "patch": stem, "dapi_cells": len(frame), "he_cells": len(keep),
                                    "candidates": int(keep.sum()), "rule_mismatch": label_mismatch, "positive_mismatch": positive_mismatch,
                                    "candidate_fraction": float(keep.mean()), "shift32_candidate_fraction_mean": float(np.mean(null)),
                                    "iou_median": float(cells.iou.median()), "distance_median": float(cells.distance.median()),
                                    "geometry_review_pass": int(cells.geometry_review_pass.sum()),
                                    "lineage_conflict": int((cells.competing_markers > 1).sum()),
                                    "local_objective": transform["local_objective"]})
                    # Every Treg candidate gets four same-extent crops, source labels only.
                    he = np.asarray(Image.open(record["he_path"]).convert("RGB"))
                    for row in cells[cells["class"] == "Treg"].to_dict("records"):
                        cx, cy = int(round(row["x"])), int(round(row["y"]))
                        x0, x1 = max(0, cx-24), min(he.shape[1], cx+25)
                        y0, y1 = max(0, cy-24), min(he.shape[0], cy+25)
                        crops = [he[y0:y1, x0:x1]] + [stack[mapping[m], y0:y1, x0:x1] for m in ["DAPI", "CD4", "FOXP3"]]
                        review.append((row, crops, (row["x"]-x0, row["y"]-y0)))
                pd.DataFrame(patches).to_csv(out / "patch_audit.csv", index=False)
                print(json.dumps({"case": case, "patch": stem, "raw_intensity_max_error": max_error, **patches[-1]}), flush=True)
            store.close()
    cells = pd.concat(all_cells, ignore_index=True)
    cells.to_csv(out / "candidate_audit.csv", index=False)
    for page, start in enumerate(range(0, len(review), 6)):
        subset = review[start:start+6]
        fig, axes = plt.subplots(len(subset), 4, figsize=(10, len(subset)*2.6), squeeze=False)
        for i, (row, crops, point) in enumerate(subset):
            for j, (crop, title) in enumerate(zip(crops, ["H&E", "DAPI", "CD4", "FOXP3"])):
                ax = axes[i, j]
                if j == 0:
                    ax.imshow(crop)
                else:
                    ax.imshow(crop, cmap="gray", vmin=0, vmax=[0, 6500, 2000, 3500][j])
                ax.scatter(*point, marker="+", c="red", s=25)
                ax.set_title(title if i == 0 else "", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            axes[i, 0].set_ylabel(f"{row['case']}\n{row['patch']}\nHE{row['he_id']} / DAPI{row['dapi_id']}\nIoU={row['iou']:.2f}", fontsize=7)
        fig.suptitle("Treg RULE CANDIDATES: raw marker evidence, NOT verified truth; + = HE center", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"treg_review_{page+1:02d}.png", dpi=130)
        plt.close(fig)
    summary = {"patches": len(patches), "candidates": len(cells), "rule_mismatches": sum(p["rule_mismatch"] for p in patches),
               "positive_mismatches": sum(p["positive_mismatch"] for p in patches), "raw_checks": raw_checks,
               "diagnostic_geometry_rule": "distance<=2, second_distance>=2*distance, IoU>=0.5, area_ratio in [0.5,2]; not validated acceptance thresholds",
               "by_class": {}, "by_case": {}, "verified_promotions": 0}
    for key, column in [("by_class", "class"), ("by_case", "case")]:
        for name, group in cells.groupby(column):
            summary[key][name] = {"n": len(group), "geometry_review_pass": int(group.geometry_review_pass.sum()),
                                  "multi_marker": int((group.competing_markers > 1).sum()),
                                  "changed_minus10pct": int(group.changed_minus10pct.sum()),
                                  "changed_plus10pct": int(group.changed_plus10pct.sum()),
                                  "changed_lower_only": int(group.changed_lower_only.sum()),
                                  "stable_single_marker": int(group.stable_single_marker.sum()),
                                  "joint_diagnostic_pass": int((group.geometry_review_pass & group.stable_single_marker).sum()),
                                  "distance_median": float(group.distance.median()), "iou_median": float(group.iou.median())}
    treg = cells[cells["class"] == "Treg"]
    summary["treg"] = {"n": len(treg), "foxp3_nuclear_positive": int(treg.foxp3_nuclear_positive.sum()),
                       "foxp3_ring_dominates": int(treg.foxp3_ring_dominates.sum()), "review_sheets": (len(review)+5)//6}
    (out / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "raw_checks"}), flush=True)


if __name__ == "__main__":
    main()
