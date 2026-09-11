#!/usr/bin/env python3
"""Audit existing CRC common-coordinate tables without promoting them to truth."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import openslide


CASES = ("2017-23800D", "2019-00538F", "2022-39941E")


def affine_error(frame, matrix):
    source = np.column_stack((frame.source_mif_x, frame.source_mif_y, np.ones(len(frame))))
    mapped = source @ np.asarray(matrix, dtype=float).T
    return np.linalg.norm(mapped[:, :2] - frame[["x", "y"]].to_numpy(), axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nfs-medical6/data/SRRSH/crc"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--sample-rows", type=int, default=20000)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or args.sample_rows < 1:
        raise ValueError("timestamp MMDD_HHMMSS and positive sample budget required")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use FewClick artifact root")
    output = args.output_root / f"crc_candidate_audit_{args.timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    base = args.root / "panel_2/registration_outputs/he_panel2_to_panel1"
    reports, candidates = [], []
    for case in CASES:
        common = base / "common_coordinate_annotations" / case
        niche_dir = base / "dual_panel_niches_40um_min60_tumor_rule" / case
        manifest = json.loads((common / "coordinate_transform_manifest.json").read_text())
        niche_manifest = json.loads((niche_dir / "manifest.json").read_text())
        with openslide.OpenSlide(manifest["fixed_panel1_he"]) as slide:
            width, height = slide.dimensions
            mpp = [float(slide.properties["openslide.mpp-x"]), float(slide.properties["openslide.mpp-y"])]
        composed = np.array(manifest["panel2_he_to_panel1_he_level0"]) @ np.array(manifest["panel2_mif_dapi_to_panel2_he_level0"])
        recorded = np.array(manifest["panel2_mif_dapi_to_panel1_he_level0_composed"])
        report = {"case_id": case, "organ_scope": "colorectal; subsite not verified",
                  "he_path": manifest["fixed_panel1_he"], "he_size": [width, height], "he_mpp_xy": mpp,
                  "manifest_size_matches": [width, height] == manifest["fixed_level0_size"],
                  "niche_mpp": niche_manifest["mpp"],
                  "niche_mpp_matches_he": bool(np.allclose(mpp, niche_manifest["mpp"], rtol=0.01)),
                  "composition_matrix_max_abs_error": float(np.max(np.abs(composed - recorded))),
                  "source_paths": {k: {"path": v, "exists": Path(v).exists()} for k, v in manifest.items()
                                   if k in ("panel1_source_cells", "panel2_source_cells", "panel2_source_geojson")},
                  "coordinate_manifest": str(common / "coordinate_transform_manifest.json"), "panels": {}}
        for panel in (1, 2):
            path = common / f"panel{panel}_cells_in_panel1_he_level0.tsv"
            frame = pd.read_csv(path, sep="\t", nrows=args.sample_rows)
            matrix_key = "panel1_mif_dapi_to_panel1_he_level0" if panel == 1 else "panel2_mif_dapi_to_panel1_he_level0_composed"
            error = affine_error(frame, manifest[matrix_key])
            xy = frame[["x", "y"]].to_numpy()
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            p = {"path": str(path), "bytes": path.stat().st_size, "sample_policy": "first rows; not spatially representative",
                 "sample_rows": len(frame), "columns": frame.columns.tolist(),
                 "manifest_total_cells": manifest[f"panel{panel}_cells"],
                 "manifest_in_bounds_cells": manifest[f"panel{panel}_cells_in_bounds"],
                 "sample_finite_xy": bool(np.isfinite(xy).all()), "sample_in_bounds_fraction": float(inside.mean()),
                 "sample_bounds_flag_agrees": bool(np.array_equal(inside, frame.in_panel1_he_bounds.to_numpy(dtype=bool))),
                 "sample_affine_error_max_px": float(error.max()), "sample_affine_error_median_px": float(np.median(error)),
                 "sample_label_counts": {str(k): int(v) for k, v in frame["class"].value_counts().items()}}
            for column in ("registered_he_patch", "local_transform"):
                if column in frame:
                    paths = frame[column].dropna().unique()[:10]
                    p[column + "_sample_access"] = [{"path": str(x), "exists": Path(x).exists()} for x in paths]
            report["panels"][str(panel)] = p
        niches = pd.read_csv(niche_dir / "offset_0/dual_panel_niches.tsv", sep="\t")
        eligible = niches[niches.dual_min60 == True].copy()
        report["primary_niche_rows"] = len(niches)
        report["eligible_primary_niches"] = len(eligible)
        report["niche_columns"] = niches.columns.tolist()
        denominator = {}
        for panel in (1, 2):
            count_cols = [f"panel{panel}_count_{name}" for name in niche_manifest["lineages"]]
            fraction_cols = [f"panel{panel}_fraction_{name}" for name in niche_manifest["lineages"]]
            counts = niches[count_cols].to_numpy(float)
            fractions = niches[fraction_cols].to_numpy(float)
            total = niches[f"n_panel{panel}"].to_numpy(float)
            full = counts / np.maximum(total[:, None], 1)
            classified = counts / np.maximum(counts.sum(1, keepdims=True), 1)
            denominator[str(panel)] = {"fraction_vs_all_cell_denominator_max_error": float(np.max(abs(fractions - full))),
                                        "fraction_vs_selected_lineages_denominator_max_error": float(np.max(abs(fractions - classified))),
                                        "meaning": "check before using as complete identity composition"}
        report["lineage_fraction_denominators"] = denominator
        geometry = pd.read_csv(niche_dir / "offset_0/niche_geometry_audit.tsv", sep="\t", nrows=1000)
        report["geometry_columns"] = geometry.columns.tolist()
        # Select spatially spread high-count review windows, NOT supervised samples.
        eligible["review_count"] = eligible[["n_panel1", "n_panel2"]].min(axis=1)
        occupied = set()
        for row in eligible.sort_values("review_count", ascending=False).to_dict("records"):
            grid = (int(row["centroid_x_um"] // 1000), int(row["centroid_y_um"] // 1000))
            if grid in occupied:
                continue
            occupied.add(grid)
            cx, cy = row["centroid_x_um"] / mpp[0], row["centroid_y_um"] / mpp[1]
            side = [int(round(128 / value)) for value in mpp]
            x, y = int(round(cx - side[0] / 2)), int(round(cy - side[1] / 2))
            if x < 0 or y < 0 or x + side[0] > width or y + side[1] > height:
                continue
            candidates.append({"case_id": case, "he_path": manifest["fixed_panel1_he"],
                               "niche_id": row["niche_id"], "offset_id": 0, "x_level0": x, "y_level0": y,
                               "width_level0": side[0], "height_level0": side[1],
                               "niche_n_panel1": row["n_panel1"], "niche_n_panel2": row["n_panel2"],
                               "status": "review_candidate_only", "registration_qc": "not_verified",
                               "identity_supervision_allowed": False, "tissue_supervision_allowed": False,
                               "note": "counts refer to entire niche, not this 128um crop; no pixel tissue truth"})
            if sum(x["case_id"] == case for x in candidates) >= 12:
                break
        reports.append(report)
        print(json.dumps({"case_id": case, "mpp": mpp, "eligible_niches": len(eligible),
                          "matrix_error": report["composition_matrix_max_abs_error"]}), flush=True)
    result = {"schema": "crc_training_candidate_audit_v1", "cases": reports,
              "scope": "3 cases; bounded cell samples, full offset0 niche tables; no image registration precision validation",
              "verified_training_manifest": False, "candidate_crops": len(candidates)}
    (output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame(candidates).to_csv(output / "review_candidates.csv", index=False)
    (output / "README.md").write_text("# CRC候选核验\n\n此目录为数据审计和人工复核入口，不是已核验训练清单。坐标矩阵重算一致不证明同一物理细胞；niche是派生邻域而非组织真值。详见audit.json与review_candidates.csv。\n")


if __name__ == "__main__":
    main()
