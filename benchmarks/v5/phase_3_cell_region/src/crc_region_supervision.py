"""Aggregate audited CRC cell labels into explicit, panel-specific region evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd


STRUCTURE_CLASSES = ("epithelial", "blood_endothelial", "lymphatic_endothelial", "sma_stromal", "fap_caf")
IMMUNE_CLASSES = ("cd4_t", "cd8_t", "treg", "b_cell", "cd163_macrophage")


def structure_class(label: str) -> str | None:
    value = str(label).removesuffix("_S100")
    if value == "Epithelial": return "epithelial"
    if value == "Blood_endothelial": return "blood_endothelial"
    if value == "Lymphatic_endothelial": return "lymphatic_endothelial"
    if value == "SMA_stromal": return "sma_stromal"
    if value in ("FAP_CAF", "FAP_SMA_CAF"): return "fap_caf"
    return None


def immune_class(label: str) -> str | None:
    value = str(label).removesuffix("_SULF1")
    return {"CD4_T": "cd4_t", "CD8_T": "cd8_t", "Treg": "treg", "B_cell": "b_cell", "CD163_Macrophage": "cd163_macrophage"}.get(value)


def aggregate_chunk(frame: pd.DataFrame, mapper, classes: tuple[str, ...], mpp_um: float, grid_um: float) -> pd.DataFrame:
    """Return counts per physical grid cell; unmapped observed cells remain in the denominator."""
    if mpp_um <= 0 or grid_um <= 0: raise ValueError("mpp_um and grid_um must be positive")
    required = {"x", "y", "class"}
    if not required.issubset(frame): raise ValueError("x, y, class columns required")
    data = frame[["x", "y", "class"]].copy()
    if not np.isfinite(data[["x", "y"]].to_numpy(float)).all(): raise ValueError("non-finite coordinates")
    data["grid_x"] = np.floor(data.x.to_numpy(float) * mpp_um / grid_um).astype(np.int64)
    data["grid_y"] = np.floor(data.y.to_numpy(float) * mpp_um / grid_um).astype(np.int64)
    observed = data.groupby(["grid_x", "grid_y"], sort=False).size().rename("observed_cells")
    data["mapped_class"] = data["class"].map(mapper)
    mapped = data[data.mapped_class.notna()]
    counts = (mapped.groupby(["grid_x", "grid_y", "mapped_class"], sort=False).size()
                    .unstack("mapped_class", fill_value=0).reindex(columns=classes, fill_value=0))
    return observed.to_frame().join(counts, how="left").fillna(0).astype(np.int64).reset_index()


def merge_aggregates(chunks: list[pd.DataFrame], classes: tuple[str, ...]) -> pd.DataFrame:
    if not chunks: raise ValueError("at least one aggregate chunk required")
    combined = pd.concat(chunks, ignore_index=True)
    return combined.groupby(["grid_x", "grid_y"], as_index=False)[["observed_cells", *classes]].sum()


def add_coverage(table: pd.DataFrame, classes: tuple[str, ...], prefix: str) -> pd.DataFrame:
    out = table.copy(); mapped = out.loc[:, classes].sum(axis=1)
    out[f"{prefix}_mapped_cells"] = mapped.astype(np.int64)
    out[f"{prefix}_coverage"] = mapped / out.observed_cells.clip(lower=1)
    out.rename(columns={"observed_cells": f"{prefix}_observed_cells", **{key: f"{prefix}_{key}_count" for key in classes}}, inplace=True)
    return out
