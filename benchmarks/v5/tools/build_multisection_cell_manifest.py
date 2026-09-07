#!/usr/bin/env python3
"""Build a deterministic, ontology-mapped cell sample manifest from audited cohorts.

This intentionally does not invent image coordinate transforms.  A cohort whose
cell coordinates cannot be shown to lie in its H&E raster stays in the manifest
with a `requires_coordinate_transform` status and is excluded from crop smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit_cell_ontology import ONTOLOGY_VERSION, REVERSE


DEFAULT_AUDIT = Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/cohorts.json")
DEFAULT_ROOT = Path("/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--timestamp")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def annotated_cells(h5ad: Path) -> pd.DataFrame:
    with h5py.File(h5ad, "r") as handle:
        obs = handle["obs"]
        index_name = obs.attrs["_index"]
        ids = [decode(value) for value in obs[index_name][:]]
        annotation = obs["annotation"]
        categories = [decode(value) for value in annotation["categories"][:]]
        codes = annotation["codes"][:]
    raw = [categories[int(code)] if code >= 0 else "" for code in codes]
    frame = pd.DataFrame({"cell_id": ids, "raw_annotation": raw})
    frame["broad_annotation"] = frame.raw_annotation.map(REVERSE)
    return frame.dropna(subset=["broad_annotation"])


def deterministic_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    groups = []
    for broad, part in frame.groupby("broad_annotation", sort=True):
        digest = int(hashlib.sha256(broad.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + digest)
        take = min(n, len(part))
        groups.append(part.iloc[np.sort(rng.choice(len(part), size=take, replace=False))])
    return pd.concat(groups, ignore_index=True)


def boundary_centers(platform: str, boundary: Path, selected: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    if platform == "Xenium":
        usecols = ["cell_id", "vertex_x", "vertex_y"]
        for chunk in pd.read_csv(boundary, usecols=usecols, chunksize=500_000, dtype={"cell_id": str}):
            hit = chunk[chunk.cell_id.isin(selected)]
            if not hit.empty:
                chunks.append(hit.rename(columns={"vertex_x": "x", "vertex_y": "y"}))
        joined = pd.concat(chunks, ignore_index=True)
        return joined.groupby("cell_id", as_index=False)[["x", "y"]].mean()
    usecols = ["fov", "cellID", "x_global_px", "y_global_px"]
    for chunk in pd.read_csv(boundary, usecols=usecols, chunksize=500_000):
        chunk["cell_id"] = chunk.cellID.astype(str) + "_" + chunk.fov.astype(str)
        hit = chunk[chunk.cell_id.isin(selected)]
        if not hit.empty:
            chunks.append(hit.rename(columns={"x_global_px": "x", "y_global_px": "y"})[["cell_id", "x", "y"]])
    joined = pd.concat(chunks, ignore_index=True)
    return joined.groupby("cell_id", as_index=False)[["x", "y"]].mean()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"multisection_cell_manifest_{stamp}"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    audit = json.loads(args.cohort_audit.read_text(encoding="utf-8"))
    cohorts = [row for row in audit if row["gate"] == "pass_contract"]
    all_rows: list[pd.DataFrame] = []
    cohort_rows = []
    for row in cohorts:
        root = Path(row["root"])
        platform, cancer = row["platform"], row["cancer"]
        cells = annotated_cells(root / "transcriptome/adata.h5ad")
        sample = deterministic_sample(cells, args.samples_per_class, args.seed)
        centers = boundary_centers(platform, root / "segmentation_mask" / ("cell_boundaries.csv" if platform == "Xenium" else "polygons.csv"), set(sample.cell_id))
        sample = sample.merge(centers, on="cell_id", how="left", validate="one_to_one")
        if sample[["x", "y"]].isna().any().any():
            raise RuntimeError(f"missing boundary centers: {platform}/{cancer}")
        status = "ready_for_he_crop_smoke" if platform == "Xenium" else "requires_coordinate_transform"
        sample = sample.assign(
            cohort_id=f"{platform}_{cancer}", platform=platform, cancer=cancer,
            he_path=str(root / "HE.tif"), h5ad_path=str(root / "transcriptome/adata.h5ad"),
            boundary_path=str(root / "segmentation_mask" / ("cell_boundaries.csv" if platform == "Xenium" else "polygons.csv")),
            coordinate_source="boundary vertex mean" if platform == "Xenium" else "polygon global-pixel mean",
            coordinate_status=status, ontology_version=ONTOLOGY_VERSION,
        )
        cols = ["cohort_id", "platform", "cancer", "cell_id", "raw_annotation", "broad_annotation", "x", "y", "coordinate_source", "coordinate_status", "ontology_version", "he_path", "h5ad_path", "boundary_path"]
        all_rows.append(sample[cols])
        cohort_rows.append({"cohort_id": f"{platform}_{cancer}", "platform": platform, "cancer": cancer, "sampled_cells": len(sample), "broad_counts": dict(sorted(Counter(sample.broad_annotation).items())), "coordinate_status": status, "coordinate_source": sample.coordinate_source.iloc[0], "he_path": str(root / "HE.tif"), "h5ad_path": str(root / "transcriptome/adata.h5ad"), "boundary_path": str(root / "segmentation_mask" / ("cell_boundaries.csv" if platform == "Xenium" else "polygons.csv"))})
    manifest = pd.concat(all_rows, ignore_index=True)
    manifest.to_csv(out / "sampled_cells.csv", index=False)
    digest = hashlib.sha256((out / "sampled_cells.csv").read_bytes()).hexdigest()
    report = {"manifest_version": "v1_0907", "ontology_version": ONTOLOGY_VERSION, "seed": args.seed, "samples_per_broad_class_per_cohort": args.samples_per_class, "source_cohort_audit": str(args.cohort_audit), "source_cohort_audit_sha256": hashlib.sha256(args.cohort_audit.read_bytes()).hexdigest(), "sampled_cells_sha256": digest, "cohorts": cohort_rows, "total_sampled_cells": len(manifest), "crop_smoke_eligible_cells": int((manifest.coordinate_status == "ready_for_he_crop_smoke").sum()), "coordinate_transform_required_cells": int((manifest.coordinate_status == "requires_coordinate_transform").sum())}
    (out / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), **{key: report[key] for key in ("total_sampled_cells", "crop_smoke_eligible_cells", "coordinate_transform_required_cells")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
