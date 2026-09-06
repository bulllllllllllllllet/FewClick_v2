#!/usr/bin/env python3
"""Run one real H&E crop through local Phikon-v2 and the v5 cell adapter."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvips
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "benchmarks/v5/phase_3_cell_region/src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from phikon_v2 import CellFeatureAdapter, PhikonV2Encoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    parser.add_argument("--he-path", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD/HE.tif"))
    parser.add_argument("--x", type=int, default=1000)
    parser.add_argument("--y", type=int, default=2000)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/backbone_smoke"))
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    if args.size != 224:
        raise ValueError("smoke contract is exactly 224x224; audit physical mpp before changing this")
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    output = args.output_root / f"phikon_v2_smoke_{stamp}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    image_vips = pyvips.Image.new_from_file(str(args.he_path), access="sequential").crop(args.x, args.y, args.size, args.size)
    array = np.ndarray(buffer=image_vips.write_to_memory(), dtype=np.uint8, shape=(args.size, args.size, image_vips.bands))[:, :, :3]
    crop = Image.fromarray(array, mode="RGB")
    encoder = PhikonV2Encoder(args.model_dir, device=device, dtype=torch.float16 if device.type == "cuda" else None)
    encoder.freeze_backbone()
    values = encoder.encode_pil([crop], device).float()
    adapter = CellFeatureAdapter().to(device).eval()
    embedding = adapter(values)
    np.save(output / "phikon_cls_1024.npy", values.cpu().numpy())
    np.save(output / "adapter_embedding_256.npy", embedding.detach().cpu().numpy())
    metadata = {
        "purpose": "real H&E -> local Phikon-v2 -> v5 adapter smoke, not a training metric",
        "model_dir": str(args.model_dir), "he_path": str(args.he_path), "crop_xy_size": [args.x, args.y, args.size],
        "device": str(device), "phikon_shape": list(values.shape), "adapter_shape": list(embedding.shape),
        "phikon_finite": bool(torch.isfinite(values).all()), "adapter_finite": bool(torch.isfinite(embedding).all()),
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), **metadata}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
