#!/usr/bin/env python3
"""Visualize measured candidate matches, without asserting registration acceptance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.probe_dir.resolve()
    if not root.is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use FewClick artifact root")
    output = root / "matching_review"
    output.mkdir(exist_ok=False)
    meta = json.loads((root / "metadata.json").read_text())
    for record in meta["records"]:
        case, stem = record["case_id"], record["patch_name"]
        image = np.array(Image.open(record["he_path"]).convert("RGB"))
        dapi = np.load(Path(meta["registration_root"]) / case / "dapi_masks" / f"{stem}.npy")
        with np.load(root / record["path"]) as data:
            xy, keep = data["he_xy"], data["candidate_valid"]
            target_ids = data["matched_dapi_id"][keep]
            target_yx = np.array(ndimage.center_of_mass(np.ones(dapi.shape), dapi, target_ids)).reshape(-1, 2)
            target_xy = target_yx[:, ::-1]
            figure, axes = plt.subplots(1, 2, figsize=(12, 6))
            for ax in axes:
                ax.imshow(image)
                ax.set_xlim(0, image.shape[1])
                ax.set_ylim(image.shape[0], 0)
                ax.axis("off")
            axes[0].set_title("Registered H&E (not original unwarped WSI)")
            axes[1].scatter(xy[:, 0], xy[:, 1], s=12, facecolors="none", edgecolors="cyan", linewidths=.6, label="H&E detections")
            axes[1].scatter(target_xy[:, 0], target_xy[:, 1], s=10, marker="x", c="yellow", linewidths=.7, label="candidate DAPI centres")
            for source, target in zip(xy[keep], target_xy):
                axes[1].plot([source[0], target[0]], [source[1], target[1]], c="yellow", linewidth=.7)
            axes[1].set_title(f"{len(xy)} HE detections; {keep.sum()} candidate labels")
            axes[1].legend(loc="lower left", fontsize=7)
            figure.suptitle(f"{case} {stem} — tentative matches, NOT verified identity truth", fontsize=11)
            figure.tight_layout()
            figure.savefig(output / f"{case}_{stem}.png", dpi=130)
            plt.close(figure)
    print(json.dumps({"panels": len(meta["records"]), "output": str(output)}))


if __name__ == "__main__":
    main()
