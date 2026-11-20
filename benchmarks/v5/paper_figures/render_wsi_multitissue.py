"""CPU rendering of selected whole-slide cases from frozen masks."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import pyvips
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def slide_arrays(case, palette):
    width, height = case["gt_width"], case["gt_height"]
    step = max(1, int(np.ceil(max(width, height) / 1100)))
    gt = pyvips.Image.new_from_file(case["gt_path"], access="random")
    small = gt.resize(1 / step, kernel="nearest")
    ys = np.minimum(np.arange(small.height) * step, height - 1)
    xs = np.minimum(np.arange(small.width) * step, width - 1)
    rgb = np.stack([
        np.frombuffer(gt.crop(0, int(y), width, 1).extract_band(0, n=3).write_to_memory(),
                      np.uint8).reshape(width, 3)[xs] for y in ys
    ])
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for color in palette:
        labels[(rgb == color["rgb"]).all(-1)] = color["id"]
    thumb = pyvips.Image.thumbnail(case["wsi_path"], small.width, height=small.height,
                                   size="force").extract_band(0, n=3)
    image = np.frombuffer(thumb.write_to_memory(), np.uint8).reshape(small.height, small.width, 3)
    return image, labels, ys, xs


def support_image(case, support):
    x, y, w, h = map(int, support["geometry"]["native_boxes"][0])
    slide = pyvips.Image.new_from_file(case["wsi_path"], access="random").extract_band(0, n=3)
    left, top, right, bottom = max(0, x), max(0, y), min(slide.width, x+w), min(slide.height, y+h)
    crop = slide.crop(left, top, right-left, bottom-top).embed(left-x, top-y, w, h,
        extend="background", background=[255, 255, 255])
    patch = np.frombuffer(crop.write_to_memory(), np.uint8).reshape(h, w, 3)
    return np.asarray(Image.fromarray(patch).resize((512, 512)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    selection = read(out / "wsi_selection.json")
    root = Path("/nfs-medical3/zyh/fewclick_v2/cvpr_research")
    cases = {c["wsi_id"]: c for c in read(root / "whole_wsi_current_1112_171010/cases.json")}
    report = read(root / "WSI_final15_closeout_1113_224245/report.json")
    reports = {}
    for path, expected_sha in report["source_report_SHA256"].items():
        path = Path(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
        value = read(path)
        reports[(value["method"], value["slide"])] = (path, value)
    palette = yaml.safe_load(Path("benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml").read_text())["data"]["class_map"]
    methods = ["HG_mix", "sansa", "sam", "sam2", "wsi", "med2d"]
    names = ["FewClick", "SANSA", "SAM", "SAM2", "WSI-SAM", "SAM-Med2D"]
    green, alpha = np.array([104, 209, 120]), 0.65
    plt.rcParams.update({"font.size": 22, "axes.titlesize": 22, "pdf.fonttype": 42})
    fig, axes = plt.subplots(len(selection["cases"]), 8, figsize=(24, 12.5), squeeze=False)
    provenance = []
    for row, selected in enumerate(selection["cases"]):
        case, target = cases[selected["slide"]], selected["target"]
        support = next(s for s in case["supports"] if s["target"] == target)
        image, labels, ys, xs = slide_arrays(case, palette)

        def overlay(mask):
            result = image.astype(float).copy()
            result[mask] = result[mask] * (1 - alpha) + green * alpha
            return result.astype(np.uint8)

        ax = axes[row]
        ax[0].imshow(support_image(case, support))
        for sign, color, marker in [("positive", "#FF0000", "+"), ("negative", "#00BFFF", "x")]:
            xy = np.array(support[sign + "_xy"]) * 512
            ax[0].scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, s=85, linewidths=2.2)
        ax[0].set_ylabel(selected["tissue"].replace(" ", "\n"), fontsize=21)
        ax[1].imshow(overlay(labels == target))
        x, y, w, h = support["geometry"]["native_boxes"][0]
        ax[1].add_patch(Rectangle((x/case["width"]*image.shape[1], y/case["height"]*image.shape[0]),
            w/case["width"]*image.shape[1], h/case["height"]*image.shape[0],
            fill=False, edgecolor="cyan", linewidth=2))
        row_receipt = dict(selected, support_id=support["support_id"],
                           positive_xy=support["positive_xy"], negative_xy=support["negative_xy"], methods=[])
        for col, (method, name) in enumerate(zip(methods, names), start=2):
            path, result = reports[(method, selected["slide"])]
            packed = np.load(path.parent / f"target_{target}/mask.npy", mmap_mode="r")
            mask = ((packed[np.ix_(ys, xs // 8)] >> (7 - xs % 8)[None, :]) & 1).astype(bool)
            assert np.array_equal(mask[:3], np.unpackbits(packed[ys[:3]], axis=1, count=case["gt_width"])[:, xs])
            entry = (next(v for v in result["targets"] if v["target"] == target)
                     if isinstance(result["targets"], list) else result["targets"][str(target)])
            score = entry["whole_WSI"]["dice"]
            if method == "HG_mix":
                assert score == selected["fewclick_dice"]
            ax[col].imshow(overlay(mask))
            ax[col].set_title((name + "\n" if row == 0 else "") + f"Dice {score*100:.1f}%")
            row_receipt["methods"].append(dict(method=method, report=str(path), whole_slide_dice=score,
                sampled_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        if row == 0:
            ax[0].set_title("Support (+/−)\n")
            ax[1].set_title("Ground truth\n+ support box")
        for panel in ax:
            panel.set_xticks([])
            panel.set_yticks([])
        provenance.append(row_receipt)
        print(f"Rendered {selected['tissue']} / {selected['slide']}", flush=True)
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.015, top=0.91,
                        wspace=0.08, hspace=0.25)
    fig.savefig(out / "figure2.png", dpi=180)
    fig.savefig(out / "figure2.pdf")
    plt.close(fig)
    (out / "render_receipt.json").write_text(json.dumps(dict(
        layout=[len(selection["cases"]), 8], overlay_rgb=green.tolist(), overlay_alpha=alpha,
        positive_color="#FF0000", negative_color="#00BFFF", selection_rule=selection["rule"],
        cases=provenance, source_reports_hash_verified=True, no_model_inference=True), ensure_ascii=False, indent=2))
    print(f"Figure 2 written to {out}", flush=True)


if __name__ == "__main__":
    main()
