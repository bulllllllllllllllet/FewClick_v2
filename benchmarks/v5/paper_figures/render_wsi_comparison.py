"""Render the fixed Figure 2 case from saved masks, without model inference."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    root = Path("/nfs-medical3/zyh/fewclick_v2/cvpr_research")
    case = read(root / "whole_wsi_current_1112_171010/cases.json")[0]
    support = case["supports"][0]
    target = support["target"]
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
    palette = yaml.safe_load(Path("benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml").read_text())["data"]["class_map"]
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for color in palette:
        labels[(rgb == color["rgb"]).all(-1)] = color["id"]
    thumb = pyvips.Image.thumbnail(case["wsi_path"], small.width, height=small.height,
                                   size="force").extract_band(0, n=3)
    image = np.frombuffer(thumb.write_to_memory(), np.uint8).reshape(small.height, small.width, 3)
    green, alpha = np.array([104, 209, 120]), 0.65

    def overlay(mask):
        result = image.astype(float).copy()
        result[mask] = result[mask] * (1 - alpha) + green * alpha
        return result.astype(np.uint8)

    plt.rcParams.update({"font.size": 18, "axes.titlesize": 18, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    ax = axes.flat
    x, y, w, h = map(int, support["geometry"]["native_boxes"][0])
    slide = pyvips.Image.new_from_file(case["wsi_path"], access="random").extract_band(0, n=3)
    left, top, right, bottom = max(0, x), max(0, y), min(slide.width, x+w), min(slide.height, y+h)
    crop = slide.crop(left, top, right-left, bottom-top).embed(left-x, top-y, w, h,
        extend="background", background=[255, 255, 255])
    patch = np.frombuffer(crop.write_to_memory(), np.uint8).reshape(h, w, 3)
    ax[0].imshow(Image.fromarray(patch).resize((512, 512)))
    for sign, color, marker in [("positive", "#FF0000", "+"), ("negative", "#00BFFF", "x")]:
        xy = np.array(support[sign + "_xy"]) * 512
        ax[0].scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, s=85, linewidths=2.2)
    ax[0].set_title("Support (+/−)\n")
    ax[1].imshow(overlay(labels == target))
    ax[1].add_patch(Rectangle((x/case["width"]*small.width, y/case["height"]*small.height),
        w/case["width"]*small.width, h/case["height"]*small.height,
        fill=False, edgecolor="cyan", linewidth=2))
    ax[1].set_title("Ground truth\n+ support box")
    contract = read(root / "paper_inference_1114_121131/contract.json")
    methods = list(contract["checkpoints"])
    names = ["FewClick", "SANSA", "SAM", "SAM2", "WSI-SAM", "SAM-Med2D"]
    assert methods == ["HG_mix", "sansa", "sam", "sam2", "wsi", "med2d"], methods
    report = read(root / "WSI_final15_closeout_1113_224245/report.json")
    provenance = []
    for k, (method, name) in enumerate(zip(methods, names)):
        path = next(Path(p) for p in report["source_report_SHA256"]
                    if "/" + method + "/" in p and "/case_00/" in p)
        packed = np.load(path.parent / f"target_{target}/mask.npy", mmap_mode="r")
        mask = ((packed[np.ix_(ys, xs // 8)] >> (7 - xs % 8)[None, :]) & 1).astype(bool)
        assert np.array_equal(mask[:3], np.unpackbits(packed[ys[:3]], axis=1, count=width)[:, xs])
        result = read(path)
        entry = (next(v for v in result["targets"] if v["target"] == target)
                 if isinstance(result["targets"], list) else result["targets"][str(target)])
        score = entry["whole_WSI"]["dice"]
        ax[k+2].imshow(overlay(mask))
        ax[k+2].set_title(f"{name}\nDice {score*100:.1f}%")
        provenance.append(dict(method=method, report=str(path), whole_slide_dice=score,
                               sampled_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    for panel in ax:
        panel.set_xticks([])
        panel.set_yticks([])
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.018, top=0.91,
                        wspace=0.065, hspace=0.29)
    fig.savefig(out / "figure2.png", dpi=180)
    fig.savefig(out / "figure2.pdf")
    plt.close(fig)
    (out / "render_receipt.json").write_text(json.dumps(dict(
        slide=case["wsi_id"], target=target, layout=[2, 4], overlay_rgb=green.tolist(),
        overlay_alpha=alpha, positive_color="#FF0000", negative_color="#00BFFF",
        click_marker_size=85, click_marker_linewidth=2.2, selection="first formal case, first support target",
        methods=provenance, no_model_inference=True), ensure_ascii=False, indent=2))
    print(f"Figure 2 written to {out}", flush=True)


if __name__ == "__main__":
    main()
