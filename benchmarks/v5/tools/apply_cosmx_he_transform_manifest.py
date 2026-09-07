#!/usr/bin/env python3
"""Create a new manifest that applies an audited CosMx global-to-H&E affine map."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--transform-audit-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/multisection_cell_manifest"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"multisection_cell_manifest_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    audit_path = args.transform_audit_dir / "transform_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["all_contract_pass"]:
        raise RuntimeError("transform audit did not pass; refusing to apply CosMx transform")
    transforms = {row["cohort_id"]: row for row in audit["cohorts"]}
    cells = pd.read_csv(args.source_manifest_dir / "sampled_cells.csv", dtype={"cell_id": str})
    for cohort, transform in transforms.items():
        selector = cells.cohort_id.eq(cohort)
        xmin, ymin, _, _ = transform["global_bounds"]
        scale_x, scale_y = transform["scale_dapi_per_he"]
        cells.loc[selector, "x"] = (cells.loc[selector, "x"] - xmin) / scale_x
        cells.loc[selector, "y"] = (cells.loc[selector, "y"] - ymin) / scale_y
        cells.loc[selector, "coordinate_source"] = "CosMx global-to-H&E affine (audited DAPI span/origin)"
        cells.loc[selector, "coordinate_status"] = "ready_for_he_crop_smoke_affine_audited"
        width, height = transform["he_size"]
        inside = (cells.loc[selector, "x"] >= 0) & (cells.loc[selector, "x"] < width) & (cells.loc[selector, "y"] >= 0) & (cells.loc[selector, "y"] < height)
        if not bool(inside.all()):
            raise RuntimeError(f"{cohort}: transformed sampled cells outside H&E")
    output.mkdir(parents=True)
    cells.to_csv(output / "sampled_cells.csv", index=False)
    source = json.loads((args.source_manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {"manifest_version": "v1_0907_cosmx_affine_audited", "source_manifest_dir": str(args.source_manifest_dir), "source_manifest_sha256": hashlib.sha256((args.source_manifest_dir / "sampled_cells.csv").read_bytes()).hexdigest(), "transform_audit_dir": str(args.transform_audit_dir), "transform_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(), "total_sampled_cells": len(cells), "crop_smoke_eligible_cells": len(cells), "coordinate_transform_required_cells": 0, "cohorts": source["cohorts"], "note": "CosMx transform passes raster-level coordinate audit and H&E crop smoke only; it is not a cell-boundary registration claim."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "eligible_cells": len(cells)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
