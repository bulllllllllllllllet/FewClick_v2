#!/usr/bin/env python3
"""Re-extract direct frozen CTransPath-768 features for CRC cached H&E cells."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file
from module.TransPath.ctran import ConvStem, ctranspath

CROP_SIZE, HALF_CROP, GAUSS_SIGMA, BG_WEIGHT, FG_WEIGHT = 128, 64, 12.0, .35, .65


def crop_region(image, cy, cx):
    h, w = image.shape[:2]
    region = image[max(0, cy-HALF_CROP):min(h, cy+HALF_CROP), max(0, cx-HALF_CROP):min(w, cx+HALF_CROP)]
    return np.pad(region, ((max(0, HALF_CROP-cy), max(0, cy+HALF_CROP-h)), (max(0, HALF_CROP-cx), max(0, cx+HALF_CROP-w)), (0, 0)), mode="edge")


def crop_mask(mask, cy, cx):
    h, w = mask.shape
    region = mask[max(0, cy-HALF_CROP):min(h, cy+HALF_CROP), max(0, cx-HALF_CROP):min(w, cx+HALF_CROP)]
    return np.pad(region, ((max(0, HALF_CROP-cy), max(0, cy+HALF_CROP-h)), (max(0, HALF_CROP-cx), max(0, cx+HALF_CROP-w))), mode="edge")


def soft_attention(image, selected):
    if not selected.any(): return image.astype(np.float32) / 255.
    blur = ndimage.gaussian_filter(selected.astype(np.float32), sigma=GAUSS_SIGMA, mode="nearest")
    return image.astype(np.float32) / 255. * (BG_WEIGHT + FG_WEIGHT * blur / (blur.max() + 1e-6))[..., None]


def load_ctranspath(path, device):
    model = ctranspath(); checkpoint = torch.load(path, map_location=device); state = checkpoint.get("model", checkpoint); adapted = {}
    for key, value in state.items():
        if "layers.0.downsample" in key: key = key.replace("layers.0.downsample", "layers.1.downsample")
        elif "layers.1.downsample" in key: key = key.replace("layers.1.downsample", "layers.2.downsample")
        elif "layers.2.downsample" in key: key = key.replace("layers.2.downsample", "layers.3.downsample")
        adapted[key] = value
    model.patch_embed = ConvStem(img_size=224, patch_size=4, in_chans=3, embed_dim=model.embed_dim)
    model.load_state_dict(adapted, strict=False)
    class Pool(torch.nn.Module):
        def __init__(self): super().__init__(); self.pool = torch.nn.AdaptiveAvgPool2d(1)
        def forward(self, value): return torch.flatten(self.pool(value.permute(0, 3, 1, 2)) if value.dim() == 4 else value, 1)
    model.head = Pool()
    return model.to(device).eval()


def resolve_he_path(record_path: str, he_root: Path | None) -> Path:
    """Map an immutable medical5-relative H&E path onto an approved mirror."""
    source = Path(record_path)
    if he_root is None:
        return source
    try:
        relative = source.relative_to("/nfs-medical5")
    except ValueError as exc:
        raise ValueError(f"cannot map non-medical5 H&E path: {source}") from exc
    mapped = he_root / relative
    if not mapped.is_file():
        raise FileNotFoundError(f"mirrored H&E file missing: {mapped}")
    return mapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--timestamp", required=True)
    p.add_argument("--max-records", type=int)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu-threads", type=int, default=48,
                   help="cap CPU preprocessing threads; set to 48 for shared-host execution")
    p.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3"))
    p.add_argument("--output-dir", type=Path, help="explicit local or NFS cache directory")
    p.add_argument("--he-root", type=Path,
                   help="read-only mirror root for metadata paths originally below /nfs-medical5")
    p.add_argument("--resume", action="store_true", help="verify and retain complete per-patch cache files")
    a = p.parse_args()
    if (not re.fullmatch(r"\d{4}_\d{6}", a.timestamp) or a.batch_size < 1 or a.cpu_threads < 1
            or (a.max_records is not None and a.max_records < 1)):
        raise ValueError("timestamp and positive budgets required")
    allowed = Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    local_cache = (ROOT / ".cache" / "semantic_phase3").resolve()
    if not (a.source_cache.resolve().is_relative_to(allowed) or a.source_cache.resolve().is_relative_to(local_cache)):
        raise ValueError("source cache must stay in FewClick NFS or local .cache/semantic_phase3")
    if a.output_dir is None and not a.output_root.resolve().is_relative_to(allowed):
        raise ValueError("NFS output root must stay in FewClick artifact root")
    if a.he_root is not None:
        mirror_root = Path("/nfs-medical6/zyh").resolve()
        if not a.he_root.resolve().is_relative_to(mirror_root):
            raise ValueError("H&E mirror root must stay below /nfs-medical6/zyh")
    meta = json.loads((a.source_cache / "metadata.json").read_text())
    records = meta["records"][:a.max_records]
    if not records:
        raise ValueError("no records selected")
    torch.set_num_threads(a.cpu_threads)
    device = torch.device(a.device)
    model = load_ctranspath(Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"), device)
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), interpolation=Image.BILINEAR), transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    out = a.output_dir.resolve() if a.output_dir else a.output_root / f"crc_ctranspath_cell_features_{a.timestamp}"
    if not (out.is_relative_to(allowed) or out.is_relative_to(local_cache)):
        raise ValueError("output must be FewClick NFS or local .cache/semantic_phase3")
    if out.exists() and not a.resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    output_records = []
    for record in records:
        source = a.source_cache / record["path"]
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"source checksum mismatch: {source.name}")
        with np.load(source, allow_pickle=False) as raw:
            values = {key: raw[key].copy() for key in raw.files}
        ids, xy, mask = values["he_cell_ids"], values["he_xy"], values["he_mask"]
        if mask.dtype.kind not in "iu" or len(ids) != len(xy) or not np.array_equal(np.unique(mask)[1:], np.sort(ids)):
            raise ValueError(f"cached mask/id geometry invalid: {source.name}")
        destination = out / source.name
        if a.resume and destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                source_hash = str(cached["source_reg64_sha256"].item()) if "source_reg64_sha256" in cached else ""
                complete = (cached["ctranspath_features"].shape == (len(ids), 768)
                            and np.isfinite(cached["ctranspath_features"]).all()
                            and np.array_equal(cached["he_cell_ids"], ids) and source_hash == record["sha256"])
            if complete:
                output_records.append({"case_id": record["case_id"], "patch_name": record["patch_name"], "path": destination.name,
                                       "source_cache": str(source), "source_cache_sha256": record["sha256"], "he_path": record["he_path"],
                                       "cells": int(len(ids)), "feature_dim": 768, "sha256": sha256_file(destination)})
                print(json.dumps({"event": "resumed", "path": destination.name}), flush=True)
                continue
        he_path = resolve_he_path(record["he_path"], a.he_root)
        image = np.asarray(Image.open(he_path).convert("RGB"))
        if image.shape[:2] != mask.shape:
            raise ValueError(f"raw H&E/cache geometry mismatch: {source.name}")
        tensors = []
        for label, (cx, cy) in zip(ids, xy):
            attended = soft_attention(crop_region(image, int(cy), int(cx)), crop_mask(mask, int(cy), int(cx)) == label)
            tensors.append(preprocess(Image.fromarray((np.clip(attended, 0, 1) * 255).astype(np.uint8))))
        with torch.inference_mode():
            parts = [model(torch.stack(tensors[start:start + a.batch_size]).to(device)).float().cpu()
                     for start in range(0, len(tensors), a.batch_size)]
        features = torch.cat(parts).numpy()
        if features.shape != (len(ids), 768) or not np.isfinite(features).all():
            raise ValueError(f"invalid CTransPath feature shape: {features.shape}")
        partial = out / f"{source.name}.partial.npz"
        np.savez_compressed(partial, ctranspath_features=features, he_cell_ids=ids, he_xy=xy,
                            source_reg64_sha256=sha256_file(source), identity_verified=values["identity_verified"],
                            candidate_class=values["candidate_class"])
        partial.replace(destination)
        result = {"case_id": record["case_id"], "patch_name": record["patch_name"], "path": destination.name,
                  "source_cache": str(source), "source_cache_sha256": sha256_file(source), "he_path": record["he_path"],
                  "resolved_he_path": str(he_path),
                  "cells": int(len(ids)), "feature_dim": 768, "sha256": sha256_file(destination)}
        output_records.append(result)
        print(json.dumps(result), flush=True)
    result_meta = {"schema": "crc_direct_ctranspath_cell_features_v1", "status": "complete", "records": output_records,
                   "feature": "frozen_ctranspath_768_from_128px_soft_attention_cell_crop", "source_cache": str(a.source_cache),
                   "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()},
                   "checkpoint_sha256": sha256_file(Path("/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth"))}
    (out / "metadata.json").write_text(json.dumps(result_meta, indent=2))
    print(json.dumps({"status": "complete", "output": str(out), "records": len(output_records)}), flush=True)


if __name__ == "__main__":
    main()
