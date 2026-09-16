#!/usr/bin/env python3
"""Create an immutable CRC cache that adopts doctor-reviewed mIF identity labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file

IDENTITY_CLASSES = ("CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage")
PROVENANCE = "expert_reviewed_mif_transferred_by_mutual_nearest_4px"


def accepted_identity_mask(candidate_valid: np.ndarray, candidate_class: np.ndarray) -> np.ndarray:
    """Rows with an existing known expert mIF identity to transfer."""
    valid = np.asarray(candidate_valid, dtype=bool)
    labels = np.asarray(candidate_class)
    if valid.ndim != 1 or labels.ndim != 1 or valid.shape != labels.shape:
        raise ValueError("candidate_valid and candidate_class must be aligned one-dimensional arrays")
    return valid & (labels >= 0) & (labels < len(IDENTITY_CLASSES))


def load_records(path: Path, allow_partial: bool) -> list[dict]:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("status") != "complete" and not (allow_partial and metadata.get("status") == "partial"):
        raise ValueError("only complete probes may be adopted unless --allow-partial-probes is explicit")
    if tuple(metadata.get("identity_classes", ())) != IDENTITY_CLASSES:
        raise ValueError("unexpected CRC identity ontology")
    records = metadata.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("feature probe has no records")
    return records


def adopt_one(source: Path, destination: Path) -> tuple[int, list[int]]:
    with np.load(source, allow_pickle=False) as raw:
        values = {name: raw[name].copy() for name in raw.files}
    for name in ("reg_features", "he_cell_ids", "he_xy", "candidate_valid", "candidate_class", "identity_verified"):
        if name not in values:
            raise ValueError(f"{source.name}: missing {name}")
    n = len(values["he_cell_ids"])
    if values["reg_features"].shape != (n, 64) or values["he_xy"].shape != (n, 2):
        raise ValueError(f"{source.name}: invalid H&E feature geometry")
    if not np.isfinite(values["reg_features"]).all():
        raise ValueError(f"{source.name}: non-finite reg64 feature")
    accepted = accepted_identity_mask(values["candidate_valid"], values["candidate_class"])
    values["identity_verified"] = accepted.astype(bool)
    values["identity_source_expert_reviewed"] = accepted.astype(bool)
    values["identity_provenance_code"] = accepted.astype(np.uint8)
    np.savez_compressed(destination, **values)
    counts = np.bincount(values["candidate_class"][accepted], minlength=len(IDENTITY_CLASSES)).astype(int).tolist()
    return int(accepted.sum()), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, action="append", required=True,
                        help="one or more feature-cache directories; repeated option preserves backward compatibility")
    parser.add_argument("--allow-partial-probes", action="store_true",
                        help="explicitly adopt completed records from partial sharded probes; skipped records remain excluded")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp must be MMDD_HHMMSS")
    probes, root = [path.resolve() for path in args.probe_dir], args.output_root.resolve()
    allowed = Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    if not all(path.is_relative_to(allowed) for path in probes) or not root.is_relative_to(allowed):
        raise ValueError("input and output must be FewClick artifact paths")
    output = root / f"crc_expert_reviewed_mif_labels_{args.timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    sources = [(probe, load_records(probe, args.allow_partial_probes)) for probe in probes]
    seen = set()
    for probe, records in sources:
        for record in records:
            key = (record.get("case_id"), record.get("patch_name"))
            if key in seen:
                raise ValueError(f"duplicate source patch across probes: {key}")
            seen.add(key)
    accepted_total, class_total, adopted_records = 0, Counter(), []
    for probe, records in sources:
        for record in records:
            source = probe / record["path"]
            if sha256_file(source) != record["sha256"]:
                raise ValueError(f"input checksum mismatch: {source.name}")
            destination = output / source.name
            accepted, counts = adopt_one(source, destination)
            new = dict(record)
            new.update({"path": destination.name, "source_probe_path": str(source), "source_probe_sha256": record["sha256"],
                        "sha256": sha256_file(destination), "accepted_identity_labels": accepted,
                        "accepted_identity_class_counts": counts, "verified_training_sample": bool(accepted),
                        "identity_label_provenance": PROVENANCE})
            adopted_records.append(new)
            accepted_total += accepted
            class_total.update(dict(zip(IDENTITY_CLASSES, counts)))
            with (output / "completed.jsonl").open("a") as stream:
                stream.write(json.dumps(new) + "\n")
            print(json.dumps({"event": "adopted", "path": destination.name, "accepted": accepted}), flush=True)
    with (output / "class_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("identity_class", "accepted_expert_reviewed_labels"))
        writer.writeheader()
        for label in IDENTITY_CLASSES:
            writer.writerow({"identity_class": label, "accepted_expert_reviewed_labels": class_total[label]})
    integrity = {"input_records": sum(len(records) for _, records in sources), "output_records": len(adopted_records), "accepted_identity_labels": accepted_total,
                 "accepted_identity_class_counts": {label: class_total[label] for label in IDENTITY_CLASSES},
                 "all_output_files_have_sha256": all(row["sha256"] for row in adopted_records)}
    (output / "integrity_audit.json").write_text(json.dumps(integrity, indent=2))
    input_probe_hashes = {str(probe): sha256_file(probe / "metadata.json") for probe, _ in sources}
    combined_probe_hash = hashlib.sha256(json.dumps(input_probe_hashes, sort_keys=True).encode()).hexdigest()
    metadata = {"schema": "crc_panel2_expert_reviewed_mif_transferred_v1", "status": "complete", "identity_classes": list(IDENTITY_CLASSES),
                "records": adopted_records, "input_probes": [str(probe) for probe, _ in sources], "input_probe_metadata_sha256": combined_probe_hash,
                "input_probe_metadata_sha256s": input_probe_hashes,
                "source_expert_reviewed": True, "identity_label_provenance": PROVENANCE,
                "identity_verified_definition": "true only for a known five-class original candidate matched by the existing mutual-nearest <=4-pixel registration rule",
                "scope": "Original mIF identity labels are accepted as doctor-reviewed reference labels per project owner confirmation. Automated checks preserve H&E/DAPI correspondence and provenance; they do not re-annotate the mIF identity.",
                "no_background_fabricated": True, "integrity_audit": integrity}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": "complete", "output": str(output), **integrity}), flush=True)


if __name__ == "__main__":
    main()
