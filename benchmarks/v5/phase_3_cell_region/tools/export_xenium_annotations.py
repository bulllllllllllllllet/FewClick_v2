#!/usr/bin/env python3
"""Export Xenium transcriptome obs/annotation without altering the source H5AD."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd


def value(item):
    return item.decode() if isinstance(item, bytes) else str(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=Path("/nfs-medical3/fyx/PKU_DATA/Xenium/COAD/transcriptome/adata.h5ad"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5ad, "r") as handle:
        obs = handle["obs"]
        ids = [value(item) for item in obs["_index"][:]]
        categories = [value(item) for item in obs["annotation/categories"][:]]
        codes = obs["annotation/codes"][:]
    labels = [categories[code] if code >= 0 else None for code in codes]
    pd.DataFrame({"cell_id": ids, "annotation": labels}).to_csv(args.output, index=False)
    print({"output": str(args.output), "cells": len(ids), "annotated": int(sum(label is not None for label in labels)), "classes": categories}, flush=True)


if __name__ == "__main__": main()
