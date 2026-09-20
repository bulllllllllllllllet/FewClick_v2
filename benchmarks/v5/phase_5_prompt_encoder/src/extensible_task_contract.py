"""Task manifests with spatial rather than patient-disjoint split guarantees."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "extensible_prompt_task_v1"
SPLITS = frozenset(("train", "val", "test"))
SUPERVISION_LEVELS = frozenset(("pixel", "cell", "region", "spot"))
PAIRING_LEVELS = frozenset(("same_cell", "same_region", "serial_section_region", "unpaired"))


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {name}")
    return value


def _footprint(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must be [x0,y0,x1,y1]")
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if not x0 < x1 or not y0 < y1:
        raise ValueError(f"{name} must have positive area")
    return x0, y0, x1, y1


def footprints_intersect(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float], buffer_um: float
) -> bool:
    """Whether buffered axis-aligned footprints overlap in a common physical coordinate system."""
    if buffer_um < 0:
        raise ValueError("buffer_um must be non-negative")
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return not (ax1 + buffer_um <= bx0 or bx1 + buffer_um <= ax0 or ay1 + buffer_um <= by0 or by1 + buffer_um <= ay0)


class ExtensibleTaskManifest:
    """Validate a task-oriented manifest without using case ID as a split group."""

    def __init__(self, spec: dict[str, Any]):
        if spec.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported extensible task manifest version")
        self.spec = spec
        self.ontology_version = _nonempty(spec.get("ontology_version"), "ontology_version")
        try:
            self.buffer_um = float(spec["split_buffer_um"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing split_buffer_um") from exc
        if self.buffer_um < 0:
            raise ValueError("split_buffer_um must be non-negative")
        rows = spec.get("samples")
        if not isinstance(rows, list) or not rows:
            raise ValueError("manifest requires samples")
        self.rows = [self._validate_row(row) for row in rows]
        self._validate_split_isolation()
        raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        self.manifest_hash = hashlib.sha256(raw).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "ExtensibleTaskManifest":
        return cls(json.loads(Path(path).read_text()))

    def _validate_row(self, row: Any) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError("sample must be an object")
        checked = dict(row)
        for field in ("sample_id", "source_id", "case_id", "section_id", "spatial_group_id", "task_id"):
            _nonempty(checked.get(field), field)
        if checked.get("split") not in SPLITS:
            raise ValueError("unknown split")
        if checked.get("supervision_level") not in SUPERVISION_LEVELS:
            raise ValueError("unknown supervision_level")
        if checked.get("pairing_level") not in PAIRING_LEVELS:
            raise ValueError("unknown pairing_level")
        _nonempty(checked.get("coordinate_system"), "coordinate_system")
        checked["source_footprint"] = _footprint(checked.get("source_footprint"), "source_footprint")
        checked["context_footprint"] = _footprint(checked.get("context_footprint"), "context_footprint")
        return checked

    def _validate_split_isolation(self) -> None:
        sample_ids: set[str] = set()
        group_splits: dict[str, str] = {}
        by_coordinate: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            if row["sample_id"] in sample_ids:
                raise ValueError("duplicate sample_id")
            sample_ids.add(row["sample_id"])
            previous = group_splits.setdefault(row["spatial_group_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("spatial_group_id appears in multiple splits")
            by_coordinate[(row["source_id"], row["coordinate_system"])].append(row)
        for rows in by_coordinate.values():
            for index, first in enumerate(rows):
                for second in rows[index + 1:]:
                    if first["split"] == second["split"]:
                        continue
                    if footprints_intersect(first["context_footprint"], second["context_footprint"], self.buffer_um):
                        raise ValueError("context footprints overlap across splits")

    def audit(self) -> dict[str, Any]:
        case_splits: dict[str, set[str]] = defaultdict(set)
        task_splits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        group_sizes: dict[str, int] = defaultdict(int)
        for row in self.rows:
            case_splits[row["case_id"]].add(row["split"])
            task_splits[row["task_id"]][row["split"]] += 1
            group_sizes[row["spatial_group_id"]] += 1
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "split_buffer_um": self.buffer_um,
            "samples": len(self.rows),
            "spatial_groups": len(group_sizes),
            "cases_crossing_splits": sorted(case for case, splits in case_splits.items() if len(splits) > 1),
            "task_split_counts": {task: dict(counts) for task, counts in sorted(task_splits.items())},
        }
