"""Explicit panel2 label vocabularies and label-aware patch sampling."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


FINE_CLASSES = ("CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage")


def fine_class(label):
    base = str(label).removesuffix("_SULF1")
    return FINE_CLASSES.index(base) if base in FINE_CLASSES else -1


def choose_balanced_patches(counts, budget):
    """Round-robin by rare-class support, penalizing repeated spatial blocks."""
    if budget < 1:
        raise ValueError("positive budget required")
    columns = list(FINE_CLASSES)
    totals = counts[columns].sum()
    classes = sorted([key for key in columns if totals[key] > 0], key=lambda key: totals[key])
    remaining = counts.copy()
    chosen, occupied = [], {}
    while len(chosen) < budget and len(remaining) and classes:
        advanced = False
        for key in classes:
            subset = remaining[remaining[key] > 0]
            if subset.empty:
                continue
            ranked = []
            for stem, row in subset.iterrows():
                match = re.fullmatch(r"x(\d+)_y(\d+)", str(stem))
                if match is None:
                    raise ValueError(f"invalid patch name {stem}")
                block = (int(match[1]) // 2048, int(match[2]) // 2048)
                score = float(row[key]) / (1 + 4 * occupied.get(block, 0))
                ranked.append((-score, str(stem), block))
            _, stem, block = min(ranked)
            occupied[block] = occupied.get(block, 0) + 1
            chosen.append(stem)
            remaining = remaining.drop(stem)
            advanced = True
            if len(chosen) >= budget:
                break
        if not advanced:
            break
    return chosen


def balanced_table(path, budget, max_dapi_cells):
    """Two streaming reads; materialize only selected patch rows."""
    totals = None
    cell_counts = None
    source_rows = 0
    for chunk in pd.read_csv(path, sep="\t", usecols=["patch_name", "class"], chunksize=200000):
        source_rows += len(chunk)
        sizes = chunk.groupby("patch_name").size()
        cell_counts = sizes if cell_counts is None else cell_counts.add(sizes, fill_value=0)
        chunk["identity"] = chunk["class"].map(fine_class)
        valid = chunk[chunk.identity >= 0]
        table = pd.crosstab(valid.patch_name, valid.identity).reindex(columns=range(5), fill_value=0)
        totals = table if totals is None else totals.add(table, fill_value=0)
    if totals is None or cell_counts is None:
        raise ValueError("empty source cell table")
    totals = totals.fillna(0).astype(np.int64)
    totals.columns = list(FINE_CLASSES)
    overall = totals.sum().to_dict()
    eligible = totals.loc[cell_counts.reindex(totals.index).between(30, max_dapi_cells)]
    selected = choose_balanced_patches(eligible, budget)
    if not selected:
        raise ValueError("no eligible class-bearing patches")
    columns = ["x", "y", "class", "patch_name", "cell_id", "registered_he_patch", "local_transform"]
    rows = []
    for chunk in pd.read_csv(path, sep="\t", usecols=columns, chunksize=200000):
        subset = chunk[chunk.patch_name.isin(selected)]
        if len(subset):
            rows.append(subset)
    combined = pd.concat(rows, ignore_index=True)
    groups = [(stem, combined[combined.patch_name == stem]) for stem in selected]
    report = {"source_rows": source_rows, "source_class_counts": {k: int(v) for k, v in overall.items()},
              "eligible_patches": len(eligible), "selected_patch_order": selected,
              "selection": "rare-class round-robin with repeated 2048px block penalty",
              "bias": "label-enriched selection; not population composition"}
    return groups, report


def quota_patch_table(path, targets, max_dapi_cells, block_px=2048):
    """Select spatially dispersed source patches until per-class mIF quotas are met."""
    if set(targets) != set(FINE_CLASSES) or any(value < 0 for value in targets.values()):
        raise ValueError("targets must be non-negative values for every fine class")
    totals = None; cell_counts = None; source_rows = 0
    for chunk in pd.read_csv(path, sep="\t", usecols=["patch_name", "class"], chunksize=200000):
        source_rows += len(chunk); sizes = chunk.groupby("patch_name").size()
        cell_counts = sizes if cell_counts is None else cell_counts.add(sizes, fill_value=0)
        chunk["identity"] = chunk["class"].map(fine_class)
        table = pd.crosstab(chunk.loc[chunk.identity >= 0, "patch_name"], chunk.loc[chunk.identity >= 0, "identity"]).reindex(columns=range(5), fill_value=0)
        totals = table if totals is None else totals.add(table, fill_value=0)
    if totals is None or cell_counts is None: raise ValueError("empty source table")
    totals = totals.fillna(0).astype(np.int64); totals.columns = list(FINE_CLASSES)
    eligible = totals.loc[cell_counts.reindex(totals.index).between(30, max_dapi_cells)].copy()
    achieved = {key: 0 for key in FINE_CLASSES}; selected=[]; occupied={}
    while len(eligible):
        deficit={key:max(0,targets[key]-achieved[key]) for key in FINE_CLASSES}
        if not any(deficit.values()): break
        ranked=[]
        for stem,row in eligible.iterrows():
            match=re.fullmatch(r"x(\d+)_y(\d+)",str(stem))
            if match is None: raise ValueError(f"invalid patch name {stem}")
            block=(int(match[1])//block_px,int(match[2])//block_px)
            gain=sum(min(int(row[key]),deficit[key])/max(1,targets[key]) for key in FINE_CLASSES if deficit[key])
            if gain: ranked.append((-gain/(1+occupied.get(block,0)),str(stem),block))
        if not ranked: break
        _,stem,block=min(ranked); row=eligible.loc[stem]
        selected.append(stem); occupied[block]=occupied.get(block,0)+1
        for key in FINE_CLASSES: achieved[key]+=int(row[key])
        eligible=eligible.drop(stem)
    report={"source_rows":source_rows,"source_class_counts":{key:int(totals[key].sum()) for key in FINE_CLASSES},"eligible_patches":int(len(cell_counts[cell_counts.between(30,max_dapi_cells)])),"targets":{key:int(targets[key]) for key in FINE_CLASSES},"achieved_source_candidates":achieved,"selected_patches":selected,"selection":"greedy quota gain with 2048px block reuse penalty","bias":"label-enriched source selection; H&E matching has not yet occurred"}
    return report
