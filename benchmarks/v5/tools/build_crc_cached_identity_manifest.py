#!/usr/bin/env python3
"""Build identity-only Phase-3 contracts directly from accepted CRC reg64 caches."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file

CLASSES = ["CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage"]
SPLITS = {"2017-23800D": "train", "2019-00538F": "val", "2022-39941E": "test"}


def cells_from_cache(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(raw["he_cell_ids"])
    height, width = raw["he_mask"].shape
    xy = raw["he_xy"].astype(np.float32) / np.array([width - 1, height - 1], np.float32)
    if not ((xy >= 0).all() and (xy <= 1).all()):
        raise ValueError("cell centers outside H&E patch")
    area = np.bincount(raw["he_mask"].ravel(), minlength=int(raw["he_mask"].max()) + 1)[raw["he_cell_ids"]]
    cells = np.zeros((n, 74), np.float32)
    cells[:, :2], cells[:, 2], cells[:, 3], cells[:, 10:] = xy, np.log1p(area), 1., raw["reg_features"]
    valid = raw["identity_verified"].astype(bool)
    allowed = np.zeros((n, len(CLASSES)), bool)
    allowed[np.arange(n)[valid], raw["candidate_class"][valid]] = True
    return cells, valid, allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--max-per-case", type=int, default=0, help="positive value is a three-case smoke budget; zero keeps all")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or args.max_per_case < 0:
        raise ValueError("timestamp and non-negative max-per-case required")
    allowed_root = Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    cache, root = args.cache_dir.resolve(), args.output_root.resolve()
    if not cache.is_relative_to(allowed_root) or not root.is_relative_to(allowed_root):
        raise ValueError("FewClick artifacts only")
    meta = json.loads((cache / "metadata.json").read_text())
    if meta.get("schema") != "crc_panel2_expert_reviewed_mif_transferred_v1" or meta.get("identity_classes") != CLASSES or not meta.get("source_expert_reviewed"):
        raise ValueError("accepted expert-reviewed CRC cache required")
    output = root / f"crc_cached_identity_manifest_{args.timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    records, seen = [], {case: 0 for case in SPLITS}
    for row in meta["records"]:
        case = row["case_id"]
        if args.max_per_case and seen[case] >= args.max_per_case:
            continue
        source = cache / row["path"]
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"checksum mismatch: {source.name}")
        with np.load(source, allow_pickle=False) as archive:
            raw = {key: archive[key].copy() for key in archive.files}
        cells, identity_valid, identity_allowed = cells_from_cache(raw)
        n = len(cells)
        name = f"{case}_{row['patch_name']}_identity.npz"
        np.savez_compressed(output / name,
                            region_tokens=np.zeros((1, 256), np.float32), cells=cells,
                            cell_valid=np.ones(n, bool), assignment_at_cells=np.ones((1, n), np.float32),
                            region_area_um2=np.ones(1, np.float32), identity_allowed=identity_allowed,
                            identity_valid=identity_valid, tissue_target=np.array([[1., 0.]], np.float32),
                            tissue_valid=np.zeros(1, bool), composition_target=np.zeros((1, len(CLASSES)), np.float32),
                            composition_valid=np.zeros(1, bool), cell_ids=raw["he_cell_ids"])
        record = {"sample_id": name.removesuffix(".npz"), "case_id": case, "organ": "colon", "split": SPLITS[case],
                  "path": name, "sha256": sha256_file(output / name), "identity_labels": int(identity_valid.sum())}
        records.append(record)
        seen[case] += 1
        print(json.dumps(record), flush=True)
    if set(row["split"] for row in records) != {"train", "val", "test"}:
        raise ValueError("contract must include all three cases")
    manifest = {"schema_version": "semantic_phase3_v1", "identity_classes": CLASSES,
                "tissue_classes": ["unavailable_0", "unavailable_1"],
                "provenance": {"feature_kind": "he_only_xcell_reg64", "sampling": "complete_detected_inventory",
                               "nuclei_class_source": "unavailable", "detector_version": "cellpose_cpsam_from_expert_cache",
                               "xcell_checkpoint_sha256": meta["input_probe_metadata_sha256"],
                               "phase2_checkpoint_sha256": "not_used_for_identity_only_cached_contract",
                               "ontology_version": "crc_panel2_fine5_expert_reviewed_v1",
                               "coordinate_version": "registered_he_dapi_mutual_nearest_le4px",
                               "label_source": "doctor_reviewed_mif_transferred_to_he",
                               "registration_qc": "cell_correspondence_verified",
                               "identity_stage_input": "cached_reg64_only; region placeholders are validation-compatible and never forwarded in identity stage"},
                "samples": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "metadata.json").write_text(json.dumps({"status": "complete", "records": len(records), "per_case": seen,
                                                        "source_cache": str(cache), "manifest_sha256": sha256_file(output / "manifest.json"),
                                                        "identity_only": True}, indent=2))
    print(json.dumps({"status": "complete", "output": str(output), "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
