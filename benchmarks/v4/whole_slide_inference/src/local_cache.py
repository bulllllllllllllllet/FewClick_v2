"""Atomic, checksum-verified local cache for immutable J5 preparation artifacts."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LocalizedPreparationCache:
    tile_index: Path
    cell_feature_manifest: Path
    region_cache_manifest: Path
    cache_dir: Path
    cache_hit: bool
    bytes_on_disk: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_with_sha256(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b""):
            writer.write(block)
            digest.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _completed_cache_dirs(root: Path, exclude: Path) -> list[Path]:
    return sorted(
        (
            item for item in root.iterdir()
            if item.is_dir() and item != exclude and not item.name.startswith(".")
            and (item / "bundle.json").is_file()
        ),
        key=lambda item: item.stat().st_mtime_ns,
    )


def _load_hit(target: Path, source_paths: dict[str, str]) -> LocalizedPreparationCache | None:
    bundle_path = target / "bundle.json"
    if not bundle_path.is_file():
        return None
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("source_paths") != source_paths:
            return None
        tile_index = target / bundle["local_files"]["tile_index"]
        cell_manifest = target / bundle["local_files"]["cell_feature_manifest"]
        region_manifest = target / bundle["local_files"]["region_cache_manifest"]
        listed = [target / value for value in bundle["local_files"].values()]
        if not all(path.is_file() for path in listed):
            return None
        os.utime(target, None)
        return LocalizedPreparationCache(
            tile_index=tile_index,
            cell_feature_manifest=cell_manifest,
            region_cache_manifest=region_manifest,
            cache_dir=target,
            cache_hit=True,
            bytes_on_disk=int(bundle["bytes_on_disk"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def materialize_preparation_cache(
    *,
    tile_index: str | Path,
    cell_feature_manifest: str | Path,
    region_cache_manifest: str | Path,
    cache_root: str | Path,
    max_bytes: int,
    min_free_bytes: int,
    cache_key_parts: tuple[str, ...] = (),
) -> LocalizedPreparationCache:
    """Copy one immutable preparation bundle locally, publishing it atomically.

    A completed local hit never stats or opens the remote source paths. Initial
    materialization validates every region shard against the preparation manifest.
    """
    if int(max_bytes) <= 0 or int(min_free_bytes) < 0:
        raise ValueError("local cache max_bytes must be positive and min_free_bytes non-negative")
    sources = {
        "tile_index": str(Path(tile_index)),
        "cell_feature_manifest": str(Path(cell_feature_manifest)),
        "region_cache_manifest": str(Path(region_cache_manifest)),
    }
    key_payload = {"source_paths": sources, "cache_key_parts": list(cache_key_parts)}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    lock_path = root / ".materialize.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for stale in root.glob(".*.partial-*"):
            if stale.is_dir():
                shutil.rmtree(stale)
        hit = _load_hit(target, sources)
        if hit is not None:
            return hit
        if target.exists():
            shutil.rmtree(target)

        remote_tile = Path(tile_index)
        remote_cell = Path(cell_feature_manifest)
        remote_region = Path(region_cache_manifest)
        manifest = pd.read_parquet(remote_region).sort_values("start").reset_index(drop=True)
        required_columns = {"shard_path", "start", "end", "rows", "sha256"}
        if not required_columns.issubset(manifest.columns):
            raise ValueError(
                f"region cache manifest misses {sorted(required_columns - set(manifest.columns))}"
            )
        shard_sources: list[Path] = []
        for value in manifest["shard_path"]:
            path = Path(value)
            shard_sources.append(path if path.is_absolute() else remote_region.parent / path)
        all_sources = [remote_tile, remote_cell, remote_region, *shard_sources]
        missing = [str(path) for path in all_sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"preparation cache source missing: {missing[0]}")
        required_bytes = sum(path.stat().st_size for path in all_sources)
        if required_bytes > int(max_bytes):
            raise OSError(
                f"preparation cache requires {required_bytes} bytes, above max_bytes={int(max_bytes)}"
            )

        existing_bytes = sum(_directory_bytes(item) for item in _completed_cache_dirs(root, target))
        candidates = _completed_cache_dirs(root, target)
        free_bytes = shutil.disk_usage(root).free
        while candidates and (
            existing_bytes + required_bytes > int(max_bytes)
            or free_bytes - required_bytes < int(min_free_bytes)
        ):
            victim = candidates.pop(0)
            victim_bytes = _directory_bytes(victim)
            shutil.rmtree(victim)
            existing_bytes -= victim_bytes
            free_bytes = shutil.disk_usage(root).free
        if existing_bytes + required_bytes > int(max_bytes):
            raise OSError("local preparation cache quota cannot accommodate this artifact")
        if shutil.disk_usage(root).free - required_bytes < int(min_free_bytes):
            raise OSError("local disk free-space floor would be violated by this artifact")

        partial = root / f".{key}.partial-{os.getpid()}"
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
        try:
            tile_name = "tile_index.parquet"
            cell_name = "cell_feature_manifest.parquet"
            region_name = "region_cache_manifest.parquet"
            source_digests = {
                "tile_index": _copy_with_sha256(remote_tile, partial / tile_name),
                "cell_feature_manifest": _copy_with_sha256(remote_cell, partial / cell_name),
                "region_cache_manifest": _sha256(remote_region),
            }
            local_shards = []
            for index, (source, expected) in enumerate(zip(shard_sources, manifest["sha256"], strict=True)):
                name = f"region_shard_{index:05d}{source.suffix}"
                actual = _copy_with_sha256(source, partial / name)
                if actual != str(expected):
                    raise ValueError(
                        f"region cache shard checksum mismatch: {source}; expected={expected} actual={actual}"
                    )
                local_shards.append(name)
            local_manifest = manifest.copy()
            local_manifest["shard_path"] = [str((target / name).resolve()) for name in local_shards]
            local_manifest.to_parquet(partial / region_name, index=False)
            local_files = {
                "tile_index": tile_name,
                "cell_feature_manifest": cell_name,
                "region_cache_manifest": region_name,
                **{f"region_shard_{index:05d}": name for index, name in enumerate(local_shards)},
            }
            bytes_on_disk = _directory_bytes(partial)
            bundle = {
                "version": 1,
                "created_unix": time.time(),
                "source_paths": sources,
                "cache_key_parts": list(cache_key_parts),
                "source_digests": source_digests,
                "local_files": local_files,
                "bytes_on_disk": bytes_on_disk,
            }
            bundle_path = partial / "bundle.json"
            bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
            with bundle_path.open("rb") as stream:
                os.fsync(stream.fileno())
            bytes_on_disk = _directory_bytes(partial)
            bundle["bytes_on_disk"] = bytes_on_disk
            bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
            with bundle_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(partial, target)
            return LocalizedPreparationCache(
                tile_index=target / tile_name,
                cell_feature_manifest=target / cell_name,
                region_cache_manifest=target / region_name,
                cache_dir=target,
                cache_hit=False,
                bytes_on_disk=bytes_on_disk,
            )
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
