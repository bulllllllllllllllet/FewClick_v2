#!/usr/bin/env python3
"""Run one real H&E crop through the local public Kaiko pathology backbone."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvips
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file


REPO = "1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms"
ARCHITECTURE = "vit_large_patch14_reg4_dinov2"
FEATURE_DIM = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/kaiko_vit_large_patch14_reg4/kaiko_0907_194700"))
    parser.add_argument("--timm-overlay", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/vendor/kaiko_timm_0907_194700"))
    parser.add_argument("--he-path", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD/HE.tif"))
    parser.add_argument("--x", type=int, default=1000)
    parser.add_argument("--y", type=int, default=2000)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/kaiko_smoke"))
    parser.add_argument("--timestamp")
    return parser.parse_args()


def load_timm(overlay: Path):
    if not overlay.is_dir():
        raise FileNotFoundError(f"Kaiko timm overlay does not exist: {overlay}")
    sys.path.insert(0, str(overlay))
    import timm
    if ARCHITECTURE not in timm.list_models("*vit_large_patch14*"):
        raise RuntimeError(f"timm {timm.__version__} lacks {ARCHITECTURE}")
    return timm


def main() -> None:
    args = parse_args()
    if args.size != 518:
        raise ValueError("Kaiko smoke contract is exactly 518px; audit physical mpp before changing it")
    weight = args.model_dir / "model.safetensors"
    config = args.model_dir / "config.json"
    if not weight.is_file() or not config.is_file():
        raise FileNotFoundError("Kaiko model directory requires verified config.json and model.safetensors")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"kaiko_pathology_smoke_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    timm = load_timm(args.timm_overlay)
    model = timm.create_model(ARCHITECTURE, pretrained=False, img_size=args.size, num_classes=0, global_pool="token")
    incompat = model.load_state_dict(load_file(str(weight), device="cpu"), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"weight mismatch: missing={incompat.missing_keys}, unexpected={incompat.unexpected_keys}")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image = pyvips.Image.new_from_file(str(args.he_path), access="sequential")
    if args.x < 0 or args.y < 0 or args.x + args.size > image.width or args.y + args.size > image.height:
        raise ValueError("requested H&E crop is outside image bounds")
    crop = image.crop(args.x, args.y, args.size, args.size)
    array = np.ndarray(buffer=crop.write_to_memory(), dtype=np.uint8, shape=(args.size, args.size, crop.bands))[:, :, :3]
    pil = Image.fromarray(array, mode="RGB")
    tensor = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).unsqueeze(0).float().div(255)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        values = model(tensor.to(device))
    if values.shape != (1, FEATURE_DIM) or not torch.isfinite(values).all():
        raise RuntimeError(f"invalid Kaiko output: {tuple(values.shape)}")
    output.mkdir(parents=True)
    np.save(output / "kaiko_cls_1024.npy", values.float().cpu().numpy())
    metadata = {
        "purpose": "real H&E -> local public Kaiko pathology backbone smoke; not a training metric",
        "repo": REPO, "architecture": ARCHITECTURE, "model_dir": str(args.model_dir), "he_path": str(args.he_path),
        "crop_xy_size": [args.x, args.y, args.size], "normalization": {"mean": MEAN, "std": STD},
        "device": str(device), "timm_version": timm.__version__, "feature_shape": list(values.shape), "finite": True,
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), **metadata}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
