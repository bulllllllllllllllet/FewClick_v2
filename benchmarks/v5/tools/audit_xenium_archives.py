#!/usr/bin/env python3
"""Read-only metadata audit of non-PKU Xenium cancer archives."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import tifffile

ROOTS = [
    ("lung", Path("/nfs-medical3/fyx/Xenium_V1_Human_Lung_Cancer_FFPE")),
    ("lung", Path("/nfs-medical3/fyx/Xenium_Prime_Human_Lung_Cancer_FFPE")),
    ("lung", Path("/nfs-medical3/fyx/Xenium_V1_humanLung_Cancer_FFPE")),
    ("breast", Path("/nfs-medical3/fyx/Xenium_V1_FFPE_Human_Breast")),
    ("breast", Path("/nfs-medical3/fyx/Xenium_FFPE_Human_Breast_Cancer_Rep1")),
    ("breast", Path("/nfs-medical3/fyx/Xenium_FFPE_Human_Breast_Cancer_Rep2")),
    ("breast_discovered_not_catalog_exact", Path("/nfs-medical3/fyx/Xenium_V1_FFPE_Preview_Human_Breast_Cancer_Sample_2")),
    ("kidney", Path("/nfs-medical3/fyx/Xenium_V1_Human_Kidney_FFPE_Protein_updated")),
    ("kidney", Path("/nfs-medical3/fyx/Xenium_V1_hKidney_cancer_section")),
    ("prostate", Path("/nfs-medical3/fyx/Xenium_Prime_Human_Prostate_FFPE")),
    ("melanoma_skin", Path("/nfs-medical3/fyx/Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def tiff_shape(path: Path) -> list[int] | None:
    try:
        with tifffile.TiffFile(path) as image: return list(image.series[0].shape)
    except Exception: return None


def archive_info(path: Path) -> dict:
    result = {"archive": str(path), "readable": False, "cells_parquet": {}, "entries": [], "error": ""}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            result["readable"] = True
            result["entries"] = [name for name in names if any(token in name.lower() for token in ("cells.parquet", "cell_boundaries", "transcripts", "analysis_summary", "cell_feature_matrix"))][:40]
            cell_entry = next((name for name in names if name.lower().endswith("cells.parquet")), "")
            if cell_entry:
                payload = archive.read(cell_entry)
                table = pq.ParquetFile(io.BytesIO(payload))
                result["cells_parquet"] = {"entry": cell_entry, "rows": int(table.metadata.num_rows), "columns": table.schema_arrow.names}
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def one(cancer: str, root: Path) -> dict:
    result = {"cancer": cancer, "root": str(root), "exists": root.is_dir(), "he_images": [], "archives": []}
    if not root.is_dir(): return result
    images = sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and ("he" in path.name.lower() or "h&e" in path.name.lower())])
    result["he_images"] = [{"name": path.name, "shape": tiff_shape(path)} for path in images]
    archives = sorted([path for path in root.glob("*.zip") if "outs" in path.name.lower() and "xe_outs" not in path.name.lower()])
    result["archives"] = [archive_info(path) for path in archives]
    return result


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"xenium_archive_audit_{stamp}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = [one(*item) for item in ROOTS]
    (out / "cohorts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Xenium archive audit", "", "| Cancer | Root | H&E images | readable archives | cells.parquet rows |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        readable = [item for item in row["archives"] if item["readable"]]
        cell_rows = sum(item["cells_parquet"].get("rows", 0) for item in readable)
        lines.append(f"| {row['cancer']} | `{Path(row['root']).name}` | {len(row['he_images'])} | {len(readable)}/{len(row['archives'])} | {cell_rows:,} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), "roots": len(rows), "existing": sum(row["exists"] for row in rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
