#!/usr/bin/env python3
"""Build a spatially isolated positive-ROI task manifest from SRRSH CRC GeoJSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v5.phase_5_prompt_encoder.src.extensible_task_contract import ExtensibleTaskManifest, SCHEMA_VERSION


CASES = ("2017-23800D", "2019-00538F", "2022-39941E")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contains(point, polygon) -> bool:
    x, y = point; inside = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def roi_polygon(path: Path):
    raw = json.loads(path.read_text())
    features = [item for item in raw.get("features", []) if item.get("properties", {}).get("classification", {}).get("name") == "Tumor"]
    if len(features) != 1 or features[0].get("geometry", {}).get("type") != "Polygon":
        raise ValueError(f"{path}: exactly one Tumor Polygon required")
    return [(float(x), float(y)) for x, y in features[0]["geometry"]["coordinates"][0][:-1]]


def build_spec(root: Path, tile_um: float, context_um: float, buffer_um: float, mpp: float):
    if min(tile_um, context_um, mpp) <= 0 or buffer_um < 0 or context_um < tile_um:
        raise ValueError("require positive tile/context/mpp, non-negative buffer, and context >= tile")
    rows = []; spacing = 2 * context_um + 2 * buffer_um
    for case in CASES:
        annotation = root / "panel_2" / "HE" / f"{case}.geojson"
        slide = root / "panel_2" / "HE" / f"{case}.svs"
        polygon = [(x * mpp, y * mpp) for x, y in roi_polygon(annotation)]
        xmin, ymin = min(x for x, _ in polygon), min(y for _, y in polygon)
        xmax, ymax = max(x for x, _ in polygon), max(y for _, y in polygon)
        x = xmin + context_um
        while x + context_um <= xmax:
            y = ymin + context_um
            while y + context_um <= ymax:
                if contains((x, y), polygon):
                    identity = f"{case}:x{round(x)}:y{round(y)}"
                    split = ("train", "val", "test")[int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 20 >= 17 and 2 or (int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 20 >= 14 and 1 or 0)]
                    half_tile = tile_um / 2; half_context = context_um / 2
                    rows.append({"sample_id": identity, "source_id": f"panel2_he:{case}", "case_id": case,
                                 "section_id": case, "spatial_group_id": identity, "task_id": "pathology.tumor_roi",
                                 "split": split, "supervision_level": "pixel", "pairing_level": "same_region",
                                 "coordinate_system": "panel2_HE_level0_um",
                                 "source_footprint": [x-half_tile, y-half_tile, x+half_tile, y+half_tile],
                                 "context_footprint": [x-half_context, y-half_context, x+half_context, y+half_context],
                                 "target_annotation_path": str(annotation), "target_annotation_sha256": sha256(annotation),
                                 "source_wsi_path": str(slide), "label_provenance": "clinician_tumor_polygon",
                                 "supervision_scope": "positive_roi_only", "valid_annotation_path": None,
                                 "negative_supervision_available": False})
                y += spacing
            x += spacing
    spec = {"schema_version": SCHEMA_VERSION, "ontology_version": "srrsh_crc_tumor_roi_v1",
            "split_buffer_um": buffer_um, "samples": rows,
            "scope": "positive tumor ROI samples only; outside-ROI pixels are not inferred as non-tumor",
            "training_ready": False}
    ExtensibleTaskManifest(spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nfs-medical6/data/SRRSH/crc"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs-medical3/zyh/fewclick_v2/extensible_prompt"))
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--tile-um", type=float, default=512.)
    parser.add_argument("--context-um", type=float, default=1024.)
    parser.add_argument("--split-buffer-um", type=float, default=128.)
    parser.add_argument("--mpp", type=float, default=.2457)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp): raise ValueError("timestamp must be MMDD_HHMMSS")
    if not args.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")): raise ValueError("output must be under FewClick artifact root")
    output = args.output_root / f"crc_tumor_roi_manifest_{args.timestamp}"; output.mkdir(parents=True, exist_ok=False)
    spec = build_spec(args.root, args.tile_um, args.context_um, args.split_buffer_um, args.mpp)
    manifest = output / "manifest.json"; manifest.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    audit = ExtensibleTaskManifest.load(manifest).audit(); audit["training_ready"] = False
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False))


if __name__ == "__main__": main()
