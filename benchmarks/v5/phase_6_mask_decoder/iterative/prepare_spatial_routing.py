"""Reindex frozen v4 cell features to the audited v5 spatial patch split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_routing(old_patch_index: Path, old_routes: dict[str, Path],
                  spatial_patch_index: Path, output: Path) -> dict:
    old = pd.read_parquet(old_patch_index)
    new = pd.read_parquet(spatial_patch_index)
    if old.patch_id.duplicated().any() or new.patch_id.duplicated().any():
        raise ValueError("patch IDs must be unique")
    pieces = []
    for split in ("train", "val", "test"):
        source = old[old.split == split].reset_index(drop=True)
        route = pd.read_parquet(old_routes[split]).sort_values("source_index").reset_index(drop=True)
        if len(source) != len(route) or not route.source_index.eq(range(len(route))).all():
            raise ValueError(f"old {split} route does not align with patch index")
        part = route.drop(columns="source_index").copy()
        part.insert(0, "patch_id", source.patch_id.astype(str).to_numpy())
        pieces.append(part)
    lookup = pd.concat(pieces, ignore_index=True).set_index("patch_id", verify_integrity=True)
    if not new.patch_id.astype(str).isin(lookup.index).all():
        raise ValueError("spatial split includes patches missing from old feature routing")
    output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split in ("train", "val", "test"):
        ids = new.loc[new.split == split, "patch_id"].astype(str)
        route = lookup.loc[ids].reset_index(drop=True)
        route.insert(0, "source_index", range(len(route)))
        route.to_parquet(output / f"cell_routing_{split}.parquet", index=False)
        counts[split] = len(route)
    audit = {"old_patch_index": str(old_patch_index), "spatial_patch_index": str(spatial_patch_index),
             "old_routes": {key: str(value) for key, value in old_routes.items()},
             "split_counts": counts, "status": "complete"}
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-patch-index", type=Path, required=True)
    parser.add_argument("--old-route-train", type=Path, required=True)
    parser.add_argument("--old-route-val", type=Path, required=True)
    parser.add_argument("--old-route-test", type=Path, required=True)
    parser.add_argument("--spatial-patch-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_routing(args.old_patch_index,
                                   {"train": args.old_route_train, "val": args.old_route_val,
                                    "test": args.old_route_test}, args.spatial_patch_index, args.output), indent=2))


if __name__ == "__main__":
    main()
