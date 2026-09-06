#!/usr/bin/env python3
"""Audit cohort section evidence and build a versioned broad cell-type ontology."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py

ONTOLOGY_VERSION = "v1_0907"
ONTOLOGY = {
    "epithelial_parenchymal": ["Epithelial", "Hepatocyte"],
    "fibroblast": ["Fibroblast"],
    "vascular_smooth_muscle": ["Endothelial", "SMC"],
    "t_cell": ["CD4T", "CD8T", "Tprolif"],
    "b_plasma_cell": ["B", "Plasma"],
    "myeloid_antigen_presenting": ["Macrophage", "Kupffer", "Monocyte", "cDC1", "cDC2", "mregDC", "pDC", "DC"],
    "innate_lymphoid": ["NK"],
    "granulocyte_mast": ["Neutrophil", "Mast"],
}
REVERSE = {label: group for group, labels in ONTOLOGY.items() for label in labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-audit", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit/cell_supervision_audit_0907_014551/cohorts.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def decode(value): return value.decode() if isinstance(value, bytes) else str(value)


def annotation_counts(path: Path) -> tuple[Counter, list[str], list[str]]:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]; group = obs["annotation"]
        categories = [decode(value) for value in group["categories"][:]]
        codes = group["codes"][:]
        counts = Counter(categories[int(code)] for code in codes if code >= 0)
        return counts, sorted(handle.get("uns", {}).keys()), sorted(handle.get("obsm", {}).keys())


def main() -> None:
    args = parse_args(); stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"cell_ontology_audit_{stamp}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    cohorts = [row for row in json.loads(args.cohort_audit.read_text()) if row["gate"] == "pass_contract"]
    rows = []
    aggregate_raw, aggregate_broad = Counter(), Counter()
    for cohort in cohorts:
        root = Path(cohort["root"]); raw, uns, obsm = annotation_counts(root / "transcriptome/adata.h5ad")
        mapped = Counter()
        for label, count in raw.items():
            mapped[REVERSE.get(label, "unmapped")] += count
        section_files = sorted(str(path.relative_to(root)) for path in root.glob("HE.*"))
        row = {"platform": cohort["platform"], "cancer": cohort["cancer"], "root": str(root), "he_section_files": section_files, "observed_section_count": len(section_files), "donor_or_sample_metadata": "not found in H5AD obs/uns; external metadata required", "h5ad_uns_keys": uns, "h5ad_obsm_keys": obsm, "raw_annotation_counts": dict(sorted(raw.items())), "broad_ontology_counts": dict(sorted(mapped.items())), "unmapped_raw_labels": sorted(set(raw) - set(REVERSE))}
        rows.append(row); aggregate_raw.update(raw); aggregate_broad.update(mapped)
    report = {"ontology_version": ONTOLOGY_VERSION, "ontology": ONTOLOGY, "cohorts": rows, "aggregate_raw_annotation_counts": dict(sorted(aggregate_raw.items())), "aggregate_broad_ontology_counts": dict(sorted(aggregate_broad.items())), "observed_he_sections": sum(row["observed_section_count"] for row in rows), "independent_donor_count": "unknown; source metadata required"}
    (out / "ontology_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Broad cell ontology audit", "", f"- Ontology version: `{ONTOLOGY_VERSION}`", f"- Contract-passing cohorts: {len(rows)}", f"- Observed H&E section files: {report['observed_he_sections']}", "- Independent donor count: **unknown**; H5AD `obs/uns` do not expose donor/sample IDs.", "", "## Cohorts", "", "| Cancer | Platform | H&E sections | Donor evidence | Unmapped labels |", "| --- | --- | ---: | --- | --- |"]
    for row in rows: lines.append(f"| {row['cancer']} | {row['platform']} | {row['observed_section_count']} | unknown | {', '.join(row['unmapped_raw_labels']) or 'none'} |")
    lines += ["", "## Aggregate broad ontology counts", ""] + [f"- `{name}`: {count:,}" for name, count in sorted(aggregate_broad.items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), "cohorts": len(rows), "sections": report["observed_he_sections"], "broad_counts": report["aggregate_broad_ontology_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
