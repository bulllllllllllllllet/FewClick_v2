#!/usr/bin/env python3
"""Download the approved Phikon-v2 checkpoint into the v5 NFS model cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="owkin/phikon-v2")
    parser.add_argument("--output-dir", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty model directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(repo_id=args.repo_id, local_dir=str(args.output_dir))
    expected = ["config.json", "preprocessor_config.json"]
    missing = [name for name in expected if not (args.output_dir / name).exists()]
    weights = sorted(args.output_dir.glob("*.safetensors")) + sorted(args.output_dir.glob("pytorch_model*.bin"))
    if missing or not weights:
        raise RuntimeError(f"incomplete Phikon-v2 download; missing={missing}, weight_files={weights}")
    print(json.dumps({"event": "complete", "repo_id": args.repo_id, "output_dir": str(args.output_dir), "resolved": resolved, "weights": [path.name for path in weights]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
