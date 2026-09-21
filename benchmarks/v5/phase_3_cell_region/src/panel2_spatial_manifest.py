"""Spatial-block splitting for panel-2 regional evidence without case-disjoint splits."""
from __future__ import annotations

import hashlib

import numpy as np

from benchmarks.v5.phase_5_prompt_encoder.src.extensible_task_contract import footprints_intersect


def deterministic_block_split(case_id, block_x, block_y, seed=20260921):
    """Assign a physical block by a stable hash; case ID is provenance, not a split unit."""
    key = f"{seed}:{case_id}:{block_x}:{block_y}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def split_and_buffer(rows, block_um=2048., buffer_um=256., seed=20260921):
    """Return retained rows and exclude every patch near a differently split patch.

    This intentionally discards boundary candidates instead of claiming that
    adjacent context windows in different splits are independent.
    """
    if block_um <= 0 or buffer_um < 0:
        raise ValueError("positive block_um and non-negative buffer_um required")
    records = []
    for row in rows:
        footprint = tuple(float(x) for x in row["footprint_um"])
        if len(footprint) != 4 or not footprint[0] < footprint[2] or not footprint[1] < footprint[3]:
            raise ValueError("valid footprint_um required")
        center_x, center_y = (footprint[0] + footprint[2]) / 2, (footprint[1] + footprint[3]) / 2
        block_x, block_y = int(np.floor(center_x / block_um)), int(np.floor(center_y / block_um))
        value = dict(row, block_x=block_x, block_y=block_y, spatial_group_id=f"{row['case_id']}:block{block_x}:{block_y}")
        value["split"] = deterministic_block_split(row["case_id"], block_x, block_y, seed)
        records.append(value)
    excluded = set()
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            second = records[j]
            if first["case_id"] != second["case_id"] or first["split"] == second["split"]:
                continue
            if footprints_intersect(tuple(first["footprint_um"]), tuple(second["footprint_um"]), buffer_um):
                excluded.add(i); excluded.add(j)
    return [value for index, value in enumerate(records) if index not in excluded], {"input": len(records), "excluded_for_cross_split_context": len(excluded), "retained": len(records) - len(excluded), "block_um": block_um, "buffer_um": buffer_um, "seed": seed}
