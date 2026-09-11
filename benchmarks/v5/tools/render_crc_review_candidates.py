#!/usr/bin/env python3
"""Render bounded H&E review windows; overlays/registration acceptance are not inferred."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openslide
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_dir.resolve()
    if not root.is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")):
        raise ValueError("use existing FewClick audit output")
    output = root / "he_review"
    output.mkdir(exist_ok=False)
    candidates = pd.read_csv(root / "review_candidates.csv")
    records = []
    for case, group in candidates.groupby("case_id"):
        group = group.head(12)
        paths = group.he_path.unique()
        if len(paths) != 1:
            raise ValueError("case has inconsistent H&E paths")
        figure, axes = plt.subplots(3, 4, figsize=(12, 9))
        with openslide.OpenSlide(paths[0]) as slide:
            for index, (ax, row) in enumerate(zip(axes.flat, group.to_dict("records"))):
                xy = (int(row["x_level0"]), int(row["y_level0"]))
                size = (int(row["width_level0"]), int(row["height_level0"]))
                if min(*xy, *size) < 0 or xy[0] + size[0] > slide.dimensions[0] or xy[1] + size[1] > slide.dimensions[1]:
                    raise ValueError("review crop outside image")
                crop = slide.read_region(xy, 0, size).convert("RGB")
                filename = f"{case}_{index:02d}.png"
                crop.save(output / filename)
                ax.imshow(crop)
                ax.set_title(f"{index:02d}: niche {int(row['niche_id'])}\nL0 {xy[0]},{xy[1]}", fontsize=8)
                ax.axis("off")
                records.append({"case_id": case, "file": filename, "niche_id": int(row["niche_id"]),
                                "source_he": paths[0], "xy_level0": xy, "size_level0": size,
                                "review_status": "unreviewed; H&E appearance only, not registration QC"})
        for ax in axes.flat[len(group):]:
            ax.axis("off")
        figure.suptitle(f"{case}: 128 um H&E windows — candidates, not ground truth", fontsize=12)
        figure.tight_layout()
        figure.savefig(output / f"{case}_contact_sheet.png", dpi=120)
        plt.close(figure)
    (output / "manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"crops": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
