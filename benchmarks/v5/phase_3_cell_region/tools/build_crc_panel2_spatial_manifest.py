#!/usr/bin/env python3
"""Build a leakage-buffered, non-case-disjoint CRC panel-2 evidence manifest."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_3_cell_region.src.panel2_spatial_manifest import split_and_buffer
from benchmarks.v5.phase_5_prompt_encoder.src.extensible_task_contract import ExtensibleTaskManifest, SCHEMA_VERSION


COUNT_COLUMNS = ("panel2_cd4_t_count", "panel2_cd8_t_count", "panel2_treg_count", "panel2_b_cell_count", "panel2_cd163_macrophage_count")
PANEL1_COUNT_COLUMNS = ("panel1_epithelial_count", "panel1_blood_endothelial_count", "panel1_lymphatic_endothelial_count", "panel1_sma_stromal_count", "panel1_fap_caf_count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--alignment-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3/crc_patch_region_alignment_0922_191000"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--block-um", type=float, default=2048.)
    parser.add_argument("--buffer-um", type=float, default=256.)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp must be MMDD_HHMMSS")
    out = args.output_root / f"crc_multpanel_spatial_manifest_{args.timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    candidates = pd.read_json(args.alignment_dir / "alignment_candidates.jsonl", lines=True)
    required = {"case_id", "patch_name", "footprint_um", "cache_path", "panel1_observed_cells", "panel1_coverage", *PANEL1_COUNT_COLUMNS, "panel2_observed_cells", "panel2_coverage", *COUNT_COLUMNS}
    if missing := required - set(candidates.columns):
        raise ValueError(f"candidate table missing {sorted(missing)}")
    grouped = []
    for (case_id, patch_name), group in candidates.groupby(["case_id", "patch_name"], sort=True):
        footprint = group.footprint_um.iloc[0]
        if any(value != footprint for value in group.footprint_um):
            raise ValueError(f"inconsistent footprint for {case_id}/{patch_name}")
        grouped.append({"case_id": case_id, "patch_name": patch_name, "footprint_um": footprint, "cache_path": group.cache_path.iloc[0],
                        "panel1_targets": [{"grid_x": int(row.grid_x), "grid_y": int(row.grid_y), "counts": [int(row[name]) for name in PANEL1_COUNT_COLUMNS], "observed_cells": int(row.panel1_observed_cells), "coverage": float(row.panel1_coverage)} for _, row in group.iterrows()], "panel2_targets": [{"grid_x": int(row.grid_x), "grid_y": int(row.grid_y), "counts": [int(row[name]) for name in COUNT_COLUMNS], "observed_cells": int(row.panel2_observed_cells), "coverage": float(row.panel2_coverage)} for _, row in group.iterrows()]})
    retained, split_audit = split_and_buffer(grouped, args.block_um, args.buffer_um, args.seed)
    samples = []
    for row in retained:
        samples.append({"sample_id": f"{row['case_id']}:{row['patch_name']}", "source_id": f"{row['case_id']}:panel1_he_level0", "case_id": row["case_id"], "section_id": f"{row['case_id']}:panel2_registered", "spatial_group_id": row["spatial_group_id"], "task_id": "observed_composition.multpanel", "split": row["split"], "supervision_level": "region", "pairing_level": "same_region", "coordinate_system": "panel1_HE_level0_um", "source_footprint": row["footprint_um"], "context_footprint": row["footprint_um"], "cache_path": row["cache_path"], "panel1_targets": row["panel1_targets"], "panel2_targets": row["panel2_targets"], "candidate_only": True})
    spec = {"schema_version": SCHEMA_VERSION, "ontology_version": "crc_multpanel_observed_composition_v1", "split_buffer_um": args.buffer_um, "samples": samples}
    manifest = ExtensibleTaskManifest(spec)
    audit = manifest.audit()
    counts = Counter(row["split"] for row in samples)
    if not all(counts[split] for split in ("train", "val", "test")):
        raise RuntimeError(f"empty split after buffered isolation: {dict(counts)}")
    case_counts = {case: dict(Counter(row["split"] for row in samples if row["case_id"] == case)) for case in sorted({row["case_id"] for row in samples})}
    (out / "manifest.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    report = {"status": "complete", "candidate_only": True, "split_audit": split_audit, "manifest_audit": audit, "split_counts": dict(counts), "case_split_counts": case_counts, "panel1_target_classes": list(PANEL1_COUNT_COLUMNS), "panel2_target_classes": list(COUNT_COLUMNS), "warning": "both panels are regional observations and this manifest is not a segmentation-label manifest"}
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(out), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
