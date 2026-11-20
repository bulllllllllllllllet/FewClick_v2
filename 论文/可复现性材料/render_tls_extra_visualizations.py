#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def panel_paths(roots: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for root in roots:
        for path in root.glob("case_*/wsi_inference_*/tls_review_panel_*.png"):
            rank = int(path.parts[-3].split("_")[1])
            if rank in result:
                raise ValueError(f"duplicate panel rank: {rank}")
            result[rank] = path
    if sorted(result) != list(range(1, 31)):
        raise ValueError(f"expected ranks 1..30, got {sorted(result)}")
    return result


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def thumbnail(path: Path, label: str, width: int = 900) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    height = int(round(image.height * width / image.width))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, image.height + 52), "white")
    canvas.paste(image, ((width - image.width) // 2, 52))
    ImageDraw.Draw(canvas).text((16, 15), label, fill="black", font=font(27))
    return canvas


def make_pages(panels: dict[int, Path], outdir: Path) -> None:
    review_dir = outdir / "TLS30_review_panels"
    review_dir.mkdir(parents=True, exist_ok=True)
    for page_index, start in enumerate(range(1, 31, 6), start=1):
        ranks = list(range(start, start + 6))
        tiles = [thumbnail(panels[rank], f"Case {rank:02d}") for rank in ranks]
        cell_width = max(tile.width for tile in tiles)
        cell_height = max(tile.height for tile in tiles)
        page = Image.new("RGB", (cell_width * 3, cell_height * 2), "white")
        for index, tile in enumerate(tiles):
            x = (index % 3) * cell_width + (cell_width - tile.width) // 2
            y = (index // 3) * cell_height + (cell_height - tile.height) // 2
            page.paste(tile, (x, y))
        page.save(review_dir / f"TLS30_review_panels_page_{page_index:02d}.jpg", quality=92, optimize=True)


def make_distribution(cases: pd.DataFrame, polygons: pd.DataFrame, outdir: Path) -> None:
    rng = np.random.default_rng(20260720)
    wsi = cases["held_out_macro_recall"].to_numpy(float)
    prompted = polygons.loc[polygons["prompted"], "recall"].to_numpy(float)
    held_out = polygons.loc[~polygons["prompted"], "recall"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].boxplot(wsi, widths=0.5, showfliers=False)
    axes[0].scatter(1 + rng.uniform(-0.12, 0.12, len(wsi)), wsi, s=22, color="#2166ac", alpha=0.85)
    axes[0].axhline(wsi.mean(), color="#b2182b", linestyle="--", linewidth=1.5, label=f"mean = {wsi.mean():.3f}")
    axes[0].set_xticks([1], ["30 WSIs"])
    axes[0].set_ylim(0, 1.03)
    axes[0].set_ylabel("Held-out TLS macro recall")
    axes[0].set_title("WSI-equal aggregation")
    axes[0].legend(frameon=False, loc="lower left")
    violin = axes[1].violinplot([prompted, held_out], showmeans=True, showextrema=False)
    for body, color in zip(violin["bodies"], ["#67a9cf", "#ef8a62"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)
    axes[1].set_xticks([1, 2], [f"Prompted\n(n={len(prompted)})", f"Held-out\n(n={len(held_out)})"])
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Per-polygon recall")
    axes[1].set_title("Polygon-equal aggregation")
    fig.savefig(outdir / "TLS30_retrieval_recall_distributions.png", dpi=250, bbox_inches="tight")
    fig.savefig(outdir / "TLS30_retrieval_recall_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def make_representatives(cases: pd.DataFrame, panels: dict[int, Path], outdir: Path) -> None:
    ordered = cases.sort_values("held_out_macro_recall").reset_index(drop=True)
    picks = [ordered.iloc[0], ordered.iloc[len(ordered) // 2], ordered.iloc[-1]]
    labels = ["Lower retrieval", "Median retrieval", "Higher retrieval"]
    tiles = []
    for descriptor, row in zip(labels, picks):
        rank = int(str(row["report_path"]).split("case_")[1].split("_")[0])
        recall = float(row["held_out_macro_recall"])
        tiles.append(thumbnail(panels[rank], f"{descriptor}: Case {rank:02d}, recall={recall:.3f}", width=1100))
    page = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
    offset = 0
    for tile in tiles:
        page.paste(tile, (offset, (page.height - tile.height) // 2))
        offset += tile.width
    page.save(outdir / "TLS30_representative_retrieval_cases.jpg", quality=93, optimize=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_csv = next(args.metrics_dir.glob("tls_metrics_by_wsi_*.csv"))
    polygon_csv = next(args.metrics_dir.glob("tls_metrics_by_polygon_*.csv"))
    cases = pd.read_csv(case_csv)
    polygons = pd.read_csv(polygon_csv)
    if len(cases) != 30 or len(polygons) != 578:
        raise ValueError(f"unexpected metrics size: cases={len(cases)} polygons={len(polygons)}")
    panels = panel_paths(args.panel_root)
    make_pages(panels, args.output_dir)
    make_distribution(cases, polygons, args.output_dir)
    make_representatives(cases, panels, args.output_dir)
    (args.output_dir / "README.md").write_text(
        "# TLS 额外可视化\n\n"
        "- `TLS30_review_panels/`：30 张外部 TLS WSI 的匿名 review-panel 图集，每页 6 张。\n"
        "- `TLS30_retrieval_recall_distributions.png/pdf`：按 WSI 等权和按多边形等权的召回率分布。\n"
        "- `TLS30_representative_retrieval_cases.jpg`：按未提示 TLS 宏召回率选择的低、中、高代表病例。\n\n"
        "Case 编号仅表示预先固定的匿名清单顺序，不包含原始病例标识。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
