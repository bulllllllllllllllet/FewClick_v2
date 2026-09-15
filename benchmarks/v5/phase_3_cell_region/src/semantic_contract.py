"""Explicit versioned manifest; no guessed class mapping or automatic data repair."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SemanticDataset(Dataset):
    def __init__(self, manifest, split):
        self.path = Path(manifest).resolve()
        self.spec = json.loads(self.path.read_text())
        if self.spec.get("schema_version") != "semantic_phase3_v1":
            raise ValueError("unsupported semantic manifest version")
        for key in ("identity_classes", "tissue_classes"):
            values = self.spec[key]
            if len(values) < 2 or len(set(values)) != len(values) or not all(isinstance(x, str) and x for x in values):
                raise ValueError(f"invalid {key}")
        provenance = self.spec["provenance"]
        if provenance.get("feature_kind") != "he_only_xcell_reg64":
            raise ValueError("first stage requires H&E-only reg64 features")
        if provenance.get("sampling") != "complete_detected_inventory":
            raise ValueError("v1 does not estimate composition from subsampled cells")
        if provenance.get("nuclei_class_source") not in ("he_prediction", "unavailable"):
            raise ValueError("GT nuclei class is prohibited in deployable inputs")
        for key in ("detector_version", "xcell_checkpoint_sha256", "phase2_checkpoint_sha256",
                    "ontology_version", "coordinate_version", "label_source", "registration_qc"):
            if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                raise ValueError(f"missing provenance {key}")
        if provenance["registration_qc"] not in ("cell_correspondence_verified", "region_correspondence_verified", "same_he_annotation"):
            raise ValueError("unverified registration cannot supervise this trainer")
        pooled_internal = provenance.get("split_protocol") == "pooled_case_overlapping_patch_internal_v1"
        seen_cases, seen_samples, seen_paths = {}, set(), set()
        for row in self.spec["samples"]:
            if row["split"] not in ("train", "val", "test"):
                raise ValueError("unknown split")
            if not row["case_id"] or not row["organ"]:
                raise ValueError("case_id and organ are required")
            sample_path = self.resolve(row)
            if row["sample_id"] in seen_samples or sample_path in seen_paths:
                raise ValueError("duplicate sample or contract path")
            seen_samples.add(row["sample_id"])
            seen_paths.add(sample_path)
            old = seen_cases.setdefault(row["case_id"], row["split"])
            if old != row["split"] and not pooled_internal:
                raise ValueError("case appears in multiple splits")
        self.rows = [row for row in self.spec["samples"] if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no samples for {split}")
        self.manifest_hash = sha256_file(self.path)

    def resolve(self, row):
        return (self.path.parent / row["path"]).resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.resolve(row)
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"contract checksum mismatch: {row['sample_id']}")
        with np.load(path, allow_pickle=False) as raw:
            values = {key: raw[key].copy() for key in (
                "region_tokens", "cells", "cell_valid", "assignment_at_cells", "region_area_um2",
                "identity_allowed", "identity_valid", "tissue_target", "tissue_valid",
                "composition_target", "composition_valid", "cell_ids")}
        r, cells = values["region_tokens"], values["cells"]
        if r.ndim != 2 or r.shape[1] != 256 or cells.ndim != 2 or cells.shape[1] != 74:
            raise ValueError("region [K,256] / cell [N,74] contract required")
        k, n = len(r), len(cells)
        c, t = len(self.spec["identity_classes"]), len(self.spec["tissue_classes"])
        shapes = {"cell_valid": (n,), "assignment_at_cells": (k, n), "region_area_um2": (k,),
                  "identity_allowed": (n, c), "identity_valid": (n,), "tissue_target": (k, t),
                  "tissue_valid": (k,), "composition_target": (k, c), "composition_valid": (k,), "cell_ids": (n,)}
        for key, shape in shapes.items():
            if values[key].shape != shape:
                raise ValueError(f"wrong shape for {key}")
        for key in ("cell_valid", "identity_allowed", "identity_valid", "tissue_valid", "composition_valid"):
            if values[key].dtype != np.bool_:
                raise ValueError(f"{key} must be bool, not inferred from numbers")
        valid = values["cell_valid"]
        if len(np.unique(values["cell_ids"][valid])) != valid.sum():
            raise ValueError("duplicate valid cell IDs")
        if values["identity_valid"].any() and self.spec["provenance"]["registration_qc"] == "region_correspondence_verified":
            raise ValueError("region registration cannot authorize cell identity labels")
        if np.any(values["identity_valid"] & (~valid | ~values["identity_allowed"].any(-1))):
            raise ValueError("identity supervision without valid cell/allowed label")
        if self.spec["provenance"]["nuclei_class_source"] == "unavailable":
            if not np.all(cells[valid, 3] == 1) or np.any(cells[valid, 4:10] != 0):
                raise ValueError("unavailable nuclei class must use unknown one-hot")
        elif np.any(cells[valid, 3:10] < 0) or not np.allclose(cells[valid, 3:10].sum(-1), 1):
            raise ValueError("predicted nuclei classes must be normalized encodings")
        if np.any(cells[valid, :2] < 0) or np.any(cells[valid, :2] > 1):
            raise ValueError("cell coordinates outside normalized patch")
        mass = values["assignment_at_cells"]
        if np.any(mass < 0) or not np.allclose(mass.sum(0)[valid], 1., atol=1e-4):
            raise ValueError("invalid cell-to-region probability mass")
        if np.any(values["region_area_um2"] < 0):
            raise ValueError("negative physical region area")
        for name in ("tissue", "composition"):
            y = values[name + "_target"][values[name + "_valid"]]
            if np.any(y < 0) or not np.allclose(y.sum(-1), 1., atol=1e-4):
                raise ValueError(f"invalid {name} target distribution")
        tensors = {}
        for key, value in values.items():
            if key == "cell_ids":
                continue
            if value.dtype != np.bool_:
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite {key}")
                value = value.astype(np.float32)
            tensors[key] = torch.from_numpy(value).unsqueeze(0)
        return {"inputs": {key: tensors[key] for key in (
                    "region_tokens", "cells", "cell_valid", "assignment_at_cells", "region_area_um2")},
                "targets": {key: value for key, value in tensors.items() if key not in (
                    "region_tokens", "cells", "cell_valid", "assignment_at_cells", "region_area_um2")},
                "case_id": row["case_id"], "sample_id": row["sample_id"]}
