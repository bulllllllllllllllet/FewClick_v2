#!/usr/bin/env python3
"""H&E-detected XCell features plus explicitly unverified mIF identity candidates."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips  # Initialize image dependencies before torch.
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from cellpose import models
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/v5/phase_3_cell_region/tools"))
from run_xenium_xcell_baseline import load_ctranspath, load_xcell, crop_region, crop_mask, soft_attention
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file
from benchmarks.v5.phase_3_cell_region.src.crc_sampling import FINE_CLASSES, fine_class, balanced_table


def relocate(value, root):
    old = Path("/home/hwx/panel2_cell_level_registration")
    path = Path(value)
    if path.is_relative_to(old):
        return root / path.relative_to(old)
    if path.is_relative_to(root):
        return path
    raise ValueError(f"unrecognized source prefix: {path}")


def broad_class(label):
    base = label.removesuffix("_SULF1")
    if base in ("CD4_T", "CD8_T", "Treg"):
        return 0
    if base == "B_cell":
        return 1
    if base == "CD163_Macrophage":
        return 2
    return -1


def centers(mask):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    yx = np.array(ndimage.center_of_mass(np.ones(mask.shape, np.uint8), mask, ids)).reshape(-1, 2)
    order = np.lexsort((yx[:, 1], yx[:, 0]))
    return ids[order], yx[order, ::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-root", type=Path, default=Path("/nfs-medical5/hwx/panel2_cell_level_registration"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--per-case", type=int, default=2)
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--selection", choices=("head", "balanced"), default="head")
    parser.add_argument("--ontology", choices=("broad3", "fine5"), default="broad3")
    parser.add_argument("--grouped-xcell", action="store_true", help="Explicitly encode all y/x-sorted cells in groups of <=255")
    parser.add_argument("--max-dapi-cells", type=int, default=180)
    parser.add_argument("--max-he-cells", type=int, default=255)
    parser.add_argument("--cases", nargs="+", choices=("2017-23800D", "2019-00538F", "2022-39941E"), default=["2017-23800D", "2019-00538F", "2022-39941E"])
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or args.per_case < 1:
        raise ValueError("valid timestamp and positive per-case budget required")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use FewClick artifact root")
    out = args.output_root / f"crc_he_feature_probe_{args.timestamp}"
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260911)
    checkpoints = {"cpsam": Path("/home/zhaoyh/.cellpose/models/cpsam"),
                   "ctranspath": Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"),
                   "xcell": Path("/home/zhaoyh/FewClick/module/checkpoint/he_model_best.pth")}
    device = torch.device(args.device)
    detector = models.CellposeModel(gpu=device.type == "cuda", pretrained_model=str(checkpoints["cpsam"]), device=device)
    ctp, xcell = load_ctranspath(checkpoints["ctranspath"], device), load_xcell(checkpoints["xcell"], device)
    ctp.requires_grad_(False)
    xcell.requires_grad_(False)
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    records = []
    selection_reports = {}
    class_names = list(FINE_CLASSES) if args.ontology == "fine5" else ["T_cell", "B_cell", "Myeloid"]
    label_fn = fine_class if args.ontology == "fine5" else broad_class
    for case in args.cases:
        folder = "cell_annotations_mif_only" if case == "2019-00538F" else "cell_annotations"
        table_path = args.registration_root / case / folder / f"{case}_registered_celltypes.tsv"
        if args.selection == "balanced":
            groups, selection_reports[case] = balanced_table(table_path, args.per_case * 3, args.max_dapi_cells)
            (out / f"selection_{case}.json").write_text(json.dumps(selection_reports[case], indent=2))
            print(json.dumps({"event": "selection_ready", "case": case, **selection_reports[case]}), flush=True)
        else:
            table = pd.read_csv(table_path, sep="\t", nrows=args.sample_rows)
            groups = table.groupby("patch_name", sort=False)
        complete = 0
        for stem, frame in groups:
            if not 30 <= len(frame) <= args.max_dapi_cells:
                continue
            he_path = relocate(frame.registered_he_patch.iloc[0], args.registration_root)
            transform_path = relocate(frame.local_transform.iloc[0], args.registration_root)
            transform = json.loads(transform_path.read_text())
            if transform["status"] != "ok" or transform.get("he_tissue_fraction", 0) < .15:
                continue
            dapi_path = args.registration_root / case / "dapi_masks" / f"{stem}.npy"
            dapi = np.load(dapi_path, allow_pickle=False)
            ids, xy = centers(dapi)
            if len(ids) != len(frame) or set(ids.tolist()) != set(frame.cell_id):
                # A head sample may end inside a patch. Do not use partial inventories.
                continue
            frame = frame.set_index("cell_id").loc[ids]
            source_xy = frame[["x", "y"]].to_numpy() - np.array([transform["x"], transform["y"]])
            error = np.linalg.norm(source_xy - xy, axis=1)
            if error.max() > 1e-3:
                raise ValueError(f"DAPI table identity/coordinate mismatch: {case}/{stem}")
            he = np.array(Image.open(he_path).convert("RGB"))
            if he.shape[:2] != dapi.shape:
                raise ValueError("registered image / DAPI geometry mismatch")
            he_mask = np.asarray(detector.eval(he, diameter=18, batch_size=1)[0], dtype=np.int32)
            he_ids, he_xy = centers(he_mask)
            maximum = args.max_he_cells if args.grouped_xcell else min(255, args.max_he_cells)
            if not 1 <= len(he_ids) <= maximum:
                with (out / "skipped.jsonl").open("a") as stream:
                    stream.write(json.dumps({"case": case, "patch": stem, "he_cells": len(he_ids), "reason": f"outside complete 1..{maximum} inventory budget"}) + "\n")
                continue
            distance, index = cKDTree(xy).query(he_xy)
            _, reverse = cKDTree(he_xy).query(xy)
            mutual = reverse[index] == np.arange(len(he_xy))
            labels = np.array([label_fn(value) for value in frame["class"]], dtype=np.int64)
            candidate = mutual & (distance <= 4.0) & (labels[index] >= 0)
            tensors = []
            for label, (cx, cy) in zip(he_ids, he_xy):
                image = soft_attention(crop_region(he, int(cy), int(cx)), crop_mask(he_mask, int(cy), int(cx)) == label)
                tensors.append(preprocess(Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))))
            with torch.inference_mode():
                raw = torch.cat([ctp(torch.stack(tensors[start:start+32]).to(device)).float() for start in range(0, len(tensors), 32)])
                outputs = []
                for start in range(0, len(raw), 255):
                    part = raw[start:start + 255]
                    padded = torch.zeros(1, 255, 768, device=device)
                    valid = torch.zeros(1, 255, device=device)
                    padded[0, :len(part)] = part
                    valid[0, :len(part)] = 1
                    _, reg, _, _ = xcell(None, padded, valid)
                    outputs.append(reg[0, :len(part)].float().cpu().numpy())
                features = np.concatenate(outputs)
            if features.shape != (len(he_ids), 64) or not np.isfinite(features).all():
                raise ValueError("invalid reg64 feature result")
            filename = f"{case}_{stem}.npz"
            np.savez_compressed(out / filename, reg_features=features, he_cell_ids=he_ids, he_xy=he_xy,
                                he_mask=he_mask, matched_dapi_id=ids[index], match_distance_px=distance,
                                candidate_class=labels[index], candidate_valid=candidate,
                                xcell_group_id=np.arange(len(he_ids)) // 255,
                                identity_verified=np.zeros(len(he_ids), bool))
            record = {"case_id": case, "patch_name": stem, "path": filename, "he_path": str(he_path),
                      "transform_path": str(transform_path), "source_table": str(table_path),
                      "dapi_cells": len(ids), "he_cells": len(he_ids), "candidate_labels": int(candidate.sum()),
                      "candidate_class_counts": np.bincount(labels[index][candidate], minlength=len(class_names)).tolist(),
                      "dapi_table_max_error_px": float(error.max()), "sha256": sha256_file(out / filename),
                      "verified_training_sample": False}
            records.append(record)
            with (out / "completed.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            complete += 1
            if complete >= args.per_case:
                break
    metadata = {"schema": "crc_panel2_he_weak_candidates_v2", "status": "complete" if len(records) == len(args.cases) * args.per_case else "partial",
                "identity_classes": class_names, "records": records,
                "selection_reports": selection_reports,
                "xcell_context": "y/x-sorted groups <=255" if args.grouped_xcell else "complete patch <=255",
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "checkpoint_sha256": {key: sha256_file(path) for key, path in checkpoints.items()},
                "registration_root": str(args.registration_root), "device": args.device,
                "feature_input": "registered H&E + newly detected H&E mask; DAPI used only for candidate targets",
                "source_rule": "registered annotation uses inclusive min AND max marker thresholds",
                "scope": "technical feature probe; mIF-assisted image registration; weak matches not verified; no accuracy claim"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"status": metadata["status"], "patches": len(records), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
