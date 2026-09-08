#!/usr/bin/env python3
"""Audit CosMx COAD cell-RNA IDs, markers, and audited H&E coordinate contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyvips


PROGRAMS = {
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "MUC1", "CEACAM5"],
    "stromal_ecm": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "COL6A1", "TAGLN"],
    "t_cell": ["CD3D", "CD3E", "TRBC1", "TRBC2", "IL7R", "LTB"],
    "b_plasma_tls": ["MS4A1", "CD79A", "CD74", "CD37", "JCHAIN", "MZB1", "CXCL13", "CCL19", "CCL21"],
    "myeloid": ["LYZ", "LST1", "TYROBP", "C1QA", "C1QB", "FCER1G"],
    "muscle": ["ACTA2", "MYH11", "TAGLN", "DES"],
    "mucus": ["MUC2", "MUC5AC", "AGR2", "SPINK4"],
    "blood": ["HBA1", "HBA2", "HBB", "ALAS2"],
    "adipose": ["ADIPOQ", "PLIN1", "FABP4", "LPL"],
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/CosMx/COAD"))
    parser.add_argument("--sampled-cells", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/cosmx_coordinate_audit/cosmx_he_transform_0907_153100/CosMx_COAD_converted_sampled_cells.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_cell_id(value: str) -> tuple[int, int]:
    cell, fov = value.rsplit("_", 1)
    return int(cell), int(fov)


def main() -> None:
    a = args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    sampled = pd.read_csv(a.sampled_cells, dtype={"cell_id": str})
    parsed = sampled.cell_id.map(parse_cell_id)
    sampled["cell_ID"] = [pair[0] for pair in parsed]
    sampled["fov"] = [pair[1] for pair in parsed]
    source = ds.dataset(a.data_root / "transcripts/tx_file.parquet", format="parquet")
    table = source.to_table(
        columns=["fov", "cell_ID", "cell", "x_global_px", "y_global_px", "target"],
        filter=ds.field("fov").isin(sorted(sampled.fov.unique().tolist())) & ds.field("cell_ID").isin(sorted(sampled.cell_ID.unique().tolist())),
    )
    tx = table.to_pandas()
    tx["cell_id"] = tx.cell_ID.astype(str) + "_" + tx.fov.astype(str)
    tx = tx[tx.cell_id.isin(set(sampled.cell_id))].copy()
    observed = set(tx.cell_id.unique())
    per_cell = tx.groupby("cell_id", as_index=False).agg(transcripts=("target", "size"), genes=("target", "nunique"))
    per_cell = sampled[["cell_id", "x", "y", "he_x", "he_y", "raw_annotation"]].merge(per_cell, on="cell_id", how="left")
    per_cell[["transcripts", "genes"]] = per_cell[["transcripts", "genes"]].fillna(0).astype(int)
    gene_cells = tx.groupby("target")["cell_id"].nunique()
    program_rows = []
    for name, markers in PROGRAMS.items():
        available = [marker for marker in markers if marker in gene_cells.index]
        hit_cells = set(tx.loc[tx.target.isin(available), "cell_id"])
        program_rows.append({"program": name, "requested_markers": "|".join(markers), "available_markers": "|".join(available), "coverage": len(available) / len(markers), "cells_nonzero_fraction": len(hit_cells) / len(sampled)})
    he = pyvips.Image.new_from_file(str(a.data_root / "HE.tif"), access="sequential")
    he_width, he_height = he.width, he.height
    # Same origin/scale relation recorded by the completed 0907 affine audit.
    scale_x = 60215 / he_width
    scale_y = 85107 / he_height
    x_he = tx.x_global_px / scale_x
    y_he = (tx.y_global_px + 4254.99983869582) / scale_y
    in_he = (x_he >= 0) & (x_he < he_width) & (y_he >= 0) & (y_he < he_height)
    metrics = {
        "sampled_cells": int(len(sampled)),
        "transcript_rows": int(len(tx)),
        "cells_with_transcripts": int(len(observed)),
        "cell_id_overlap": float(len(observed) / len(sampled)),
        "he_size": [he_width, he_height],
        "global_to_he_in_bounds_fraction": float(in_he.mean()) if len(tx) else 0.0,
        "programs": program_rows,
        "acceptance": {"cell_id_overlap_at_least_0_95": len(observed) / len(sampled) >= 0.95, "all_transcripts_map_inside_he": bool(in_he.all()), "has_two_marker_program": any(row["coverage"] * len(PROGRAMS[row["program"]]) >= 2 and row["cells_nonzero_fraction"] > 0 for row in program_rows)},
    }
    a.output_dir.mkdir(parents=True)
    per_cell.to_csv(a.output_dir / "cell_rna_overlap.csv", index=False)
    pd.DataFrame(program_rows).to_csv(a.output_dir / "marker_program_audit.csv", index=False)
    (a.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(a.output_dir), **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
