#!/usr/bin/env python3
"""Boundedly audit every server path listed in the FewClick dataset catalogue."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


KEYWORDS = {
    "he_or_pathology": re.compile(r"(^|[_/.-])(he|h&e|histology|wsi)([_/.-]|$)|\.(svs|ndpi|mrxs)$", re.I),
    "rna_or_expression": re.compile(r"transcript|rna|gene|expression|feature_matrix|matrix\.mtx|adata\.h5ad", re.I),
    "protein": re.compile(r"protein|codex|cite", re.I),
    "boundary_or_segmentation": re.compile(r"cell_boundar|nucleus_boundar|polygon|segmentation", re.I),
    "annotation_or_label": re.compile(r"annotation|label|cluster|cell[s_].*\.(csv|parquet)", re.I),
    "image": re.compile(r"\.(tif|tiff|svs|ndpi|ome\.tif|png|jpg|jpeg)$", re.I),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("docs/数据集情况/【7】基因-病理-疾病相关数据.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/dataset_audit"))
    parser.add_argument("--timestamp")
    parser.add_argument("--max-entries", type=int, default=300)
    return parser.parse_args()


def clean(value: str) -> str:
    return value.strip().replace("\n", " ")


def catalogue_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    records = []
    for row_number, row in enumerate(rows[2:], start=3):
        if len(row) <= 8 or not clean(row[8]):
            continue
        padded = row + [""] * max(0, 26 - len(row))
        records.append({
            "catalog_row": str(row_number), "organ_or_cancer": clean(padded[0]), "provider": clean(padded[1]),
            "task": clean(padded[3]), "pathology_claim": clean(padded[4]), "genomics_claim": clean(padded[5]),
            "omics_claim": clean(padded[6]), "server_path": clean(padded[8]), "catalog_pairing": clean(padded[9]),
            "catalog_annotation": clean(padded[10]), "spatial_resolution": clean(padded[14]),
            "tissue_type": clean(padded[18]), "stain": clean(padded[19]), "catalog_quantity": clean(padded[20]),
            "catalog_data_pairing": clean(padded[21]), "catalog_labels": clean(padded[22]), "source": clean(padded[16]),
        })
    return records


def bounded_names(root: Path, maximum: int) -> tuple[list[str], int, bool]:
    names: list[str] = []
    seen = 0
    truncated = False
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue and len(names) < maximum:
        current, depth = queue.pop(0)
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name.lower())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            seen += 1
            relative = str(entry.relative_to(root))
            names.append(relative + ("/" if entry.is_dir() else ""))
            if entry.is_dir() and depth < 1:
                queue.append((entry, depth + 1))
            if len(names) >= maximum:
                truncated = bool(queue) or seen < sum(1 for _ in entries)
                break
    return names, seen, truncated


def audit(record: dict[str, str], maximum: int) -> dict[str, object]:
    target = Path(record["server_path"])
    value: dict[str, object] = dict(record)
    try:
        if not target.exists():
            value.update(status="missing", path_kind="", platform_hint="", visible_entries=0, truncated=False, modality_hints="", matched_entries="", error="")
            return value
        if target.is_file():
            names = [target.name]
            kind = "file"
        elif target.is_dir():
            names, _, truncated = bounded_names(target, maximum)
            kind = "directory"
            value["truncated"] = truncated
        else:
            names, kind = [], "other"
        lower_path = record["server_path"].lower()
        platform = "xenium" if "xenium" in lower_path else "cosmx" if "cosmx" in lower_path else "visium" if "visium" in lower_path else "stereo-seq" if "stereo" in lower_path else "other"
        if platform == "other" and any("outs.zip" in item.lower() for item in names): platform = "archived_xenium_candidate"
        hints = [name for name, pattern in KEYWORDS.items() if any(pattern.search(item.rstrip("/")) for item in names)]
        matched = [item for item in names if any(pattern.search(item.rstrip("/")) for pattern in KEYWORDS.values())]
        value.update(status="exists", path_kind=kind, platform_hint=platform, visible_entries=len(names), truncated=value.get("truncated", False), modality_hints=";".join(hints), matched_entries=" | ".join(matched[:40]), error="")
    except (OSError, PermissionError) as exc:
        value.update(status="error", path_kind="", platform_hint="", visible_entries=0, truncated=False, modality_hints="", matched_entries="", error=repr(exc))
    return value


def markdown(rows: list[dict[str, object]], stamp: str) -> str:
    states = Counter(str(row["status"]) for row in rows)
    with_pathology = [row for row in rows if row["status"] == "exists" and "he_or_pathology" in str(row["modality_hints"])]
    cell_platforms = {"xenium", "cosmx"}
    with_cell_rna = [row for row in rows if row["status"] == "exists" and str(row.get("platform_hint", "")) in cell_platforms and all(key in str(row["modality_hints"]) for key in ("rna_or_expression", "boundary_or_segmentation"))]
    lines = [
        "# 数据集目录全量审计", "", f"- 时间戳：`{stamp}`", f"- CSV 非空路径记录：{len(rows)}（路径可重复）",
        f"- 状态：存在 {states['exists']}；缺失 {states['missing']}；读取错误 {states['error']}。",
        f"- 文件名线索含 H&E/病理图：{len(with_pathology)} 条；Xenium/CosMx 中同时含 RNA 与边界/分割文件名线索：{len(with_cell_rna)} 条。", "",
        "本报告是有界目录扫描，`modality_hints` 仅依据文件名。Visium/Stereo-seq 的 segmentation 文件不代表单细胞边界；所有线索均不能替代读取数据字段、坐标配准、sample/donor 数和标签质量审计。", "",
        "## 优先进入下一轮单细胞监督审计的候选", "",
        "| CSV 行 | 癌种/器官 | 平台 | 路径 | 文件名线索 |", "| ---: | --- | --- | --- | --- |",
    ]
    for row in with_cell_rna:
        lines.append(f"| {row['catalog_row']} | {row['organ_or_cancer'] or '（继承上一行）'} | {row['platform_hint']} | `{row['server_path']}` | {row['modality_hints']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stamp = args.timestamp or datetime.now().strftime("%m%d_%H%M%S")
    out = args.output_root / f"dataset_catalog_audit_{stamp}"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = [audit(item, args.max_entries) for item in catalogue_rows(args.catalog)]
    fields = list(rows[0]) if rows else []
    with (out / "audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out / "summary.md").write_text(markdown(rows, stamp), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({"timestamp": stamp, "catalog": str(args.catalog), "records": len(rows), "max_entries": args.max_entries, "status_counts": Counter(str(row["status"]) for row in rows)}, indent=2, default=dict), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out), "records": len(rows), "status_counts": Counter(str(row["status"]) for row in rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
