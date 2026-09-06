#!/usr/bin/env python3
"""Deep read-only contract audit for the six expanded PKU spatial cohorts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import tifffile

COHORTS = [(platform, cancer, Path(f"/nfs-medical3/fyx/PKU_DATA/{platform}/{cancer}")) for platform in ("Xenium", "CosMx") for cancer in ("COAD", "HCC", "OV")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def decoded(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def h5ad_info(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_name = decoded(obs.attrs["_index"])
        ids = {decoded(value) for value in obs[index_name][:]}
        annotation = {"field": "", "annotated": 0, "categories": {}, "coverage": 0.0}
        if "annotation" in obs:
            item = obs["annotation"]
            if isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
                categories = [decoded(value) for value in item["categories"][:]]
                codes = item["codes"][:]
                counts = Counter(categories[int(code)] for code in codes if code >= 0)
                annotation = {"field": "annotation", "annotated": int((codes >= 0).sum()), "categories": dict(sorted(counts.items())), "coverage": float((codes >= 0).mean())}
        obsm = {key: list(handle["obsm"][key].shape) for key in handle.get("obsm", {})}
        uns_keys = sorted(handle.get("uns", {}).keys())
        return {"obs": len(ids), "var": int(handle["var"].attrs.get("_index", "") != "") and len(handle["var"][decoded(handle["var"].attrs["_index"])]) or 0, "obs_index_field": index_name, "ids": ids, "annotation": annotation, "obsm": obsm, "uns_keys": uns_keys}


def csv_ids(path: Path, platform: str) -> tuple[str, set[str], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle); header = next(reader)
    if platform == "CosMx" and "fov" in header and "cellID" in header:
        values: set[str] = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("fov") and row.get("cellID"):
                    values.add(f"{row['cellID']}_{row['fov']}")
        return "cellID+fov", values, header
    candidates = [name for name in header if name.lower().replace(" ", "_") in {"cell_id", "cellid", "cell_id_", "cell"}]
    if not candidates:
        candidates = [name for name in header if "cell" in name.lower() and "id" in name.lower()]
    if not candidates: raise RuntimeError(f"no cell ID column in {path}: {header}")
    field = candidates[0]; values: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(field): values.add(str(row[field]))
    return field, values, header


def tiff_shape(path: Path) -> list[int]:
    with tifffile.TiffFile(path) as image:
        return list(image.series[0].shape)


def parquet_info(path: Path) -> dict:
    try:
        file = pq.ParquetFile(path)
    except Exception as exc:
        return {"rows": 0, "columns": [], "cell_id_column": "", "coordinate_columns": [], "read_error": repr(exc)}
    columns = file.schema_arrow.names
    lower = {name.lower(): name for name in columns}
    cell = next((original for name, original in lower.items() if name in {"cell_id", "cellid", "cell"}), "")
    coord = [name for name in columns if any(token in name.lower() for token in ("x_location", "y_location", "x_coord", "y_coord", "center_x", "center_y", "x", "y"))]
    return {"rows": int(file.metadata.num_rows), "columns": columns, "cell_id_column": cell, "coordinate_columns": coord, "read_error": ""}


def one(platform: str, cancer: str, root: Path) -> dict:
    he = root / "HE.tif"; h5ad = root / "transcriptome/adata.h5ad"; protein = root / "proteome/adata_codex.h5ad"
    boundary = root / ("segmentation_mask/cell_boundaries.csv" if platform == "Xenium" else "segmentation_mask/polygons.csv")
    transcript = root / ("transcripts/transcripts.parquet" if platform == "Xenium" else "transcripts/tx_file.parquet")
    required = [he, h5ad, boundary, transcript]
    result = {"platform": platform, "cancer": cancer, "root": str(root), "required_exists": {str(path.relative_to(root)): path.is_file() for path in required}, "protein_h5ad_exists": protein.is_file()}
    if not all(result["required_exists"].values()):
        result["gate"] = "fail_missing_required"; return result
    info = h5ad_info(h5ad); boundary_field, ids, header = csv_ids(boundary, platform); transcript_info = parquet_info(transcript)
    overlap = len(info["ids"] & ids); boundary_coverage = overlap / max(len(ids), 1); obs_coverage = overlap / max(len(info["ids"]), 1)
    result.update({"he_shape": tiff_shape(he), "h5ad": {key: value for key, value in info.items() if key != "ids"}, "boundary": {"path": str(boundary.relative_to(root)), "cell_id_field": boundary_field, "unique_cell_ids": len(ids), "columns": header}, "transcripts": transcript_info, "id_overlap": {"count": overlap, "fraction_of_boundary": boundary_coverage, "fraction_of_h5ad_obs": obs_coverage}})
    fields_ok = bool(not transcript_info["read_error"] and transcript_info["cell_id_column"] and len(transcript_info["coordinate_columns"]) >= 2)
    result["gate"] = "pass_contract" if boundary_coverage >= .95 and fields_ok and info["annotation"]["coverage"] > 0 else "review_alignment_or_annotation"
    return result


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"cell_supervision_audit_{stamp}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    results = [one(*cohort) for cohort in COHORTS]
    (out / "cohorts.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# Cell supervision cohort audit", "", "| Platform | Cancer | H5AD obs | Boundary cells | overlap / boundary | annotation coverage | Transcript rows | Gate |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in results:
        h5 = item.get("h5ad", {}); boundary = item.get("boundary", {}); overlap = item.get("id_overlap", {})
        lines.append(f"| {item['platform']} | {item['cancer']} | {h5.get('obs','')} | {boundary.get('unique_cell_ids','')} | {overlap.get('fraction_of_boundary',0):.4f} | {h5.get('annotation',{}).get('coverage',0):.4f} | {item.get('transcripts',{}).get('rows','')} | {item['gate']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), "gates": Counter(item["gate"] for item in results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
