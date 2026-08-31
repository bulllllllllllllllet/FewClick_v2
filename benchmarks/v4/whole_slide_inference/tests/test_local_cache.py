from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarks.v4.whole_slide_inference.src.local_cache import materialize_preparation_cache


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalPreparationCacheTest(unittest.TestCase):
    def _sources(self, root: Path, *, bad_digest: bool = False) -> tuple[Path, Path, Path]:
        remote = root / "remote"
        remote.mkdir()
        tile = remote / "tile_index.parquet"
        cell = remote / "feature_index.parquet"
        tile.write_bytes(b"tile-index")
        cell.write_bytes(b"cell-manifest")
        shard_paths = []
        rows = []
        for index in range(2):
            shard = remote / f"region_{index}.pt"
            shard.write_bytes(f"region-shard-{index}".encode())
            shard_paths.append(shard)
            rows.append({
                "shard_path": str(shard), "start": index, "end": index + 1,
                "rows": 1, "sha256": "0" * 64 if bad_digest and index == 1 else sha256(shard),
            })
        manifest = remote / "region_cache_index.parquet"
        pd.DataFrame(rows).to_parquet(manifest, index=False)
        return tile, cell, manifest

    def test_second_hit_does_not_require_remote_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tile, cell, manifest = self._sources(root)
            cache_root = root / "local"
            first = materialize_preparation_cache(
                tile_index=tile, cell_feature_manifest=cell, region_cache_manifest=manifest,
                cache_root=cache_root, max_bytes=10_000_000, min_free_bytes=0,
                cache_key_parts=("model-v1",),
            )
            self.assertFalse(first.cache_hit)
            self.assertTrue(first.region_cache_manifest.is_file())
            local_manifest = pd.read_parquet(first.region_cache_manifest)
            self.assertTrue(all(Path(path).is_file() for path in local_manifest["shard_path"]))
            (root / "remote").rename(root / "remote-offline")
            second = materialize_preparation_cache(
                tile_index=tile, cell_feature_manifest=cell, region_cache_manifest=manifest,
                cache_root=cache_root, max_bytes=10_000_000, min_free_bytes=0,
                cache_key_parts=("model-v1",),
            )
            self.assertTrue(second.cache_hit)
            self.assertEqual(first.cache_dir, second.cache_dir)

    def test_bad_shard_checksum_is_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tile, cell, manifest = self._sources(root, bad_digest=True)
            cache_root = root / "local"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                materialize_preparation_cache(
                    tile_index=tile, cell_feature_manifest=cell, region_cache_manifest=manifest,
                    cache_root=cache_root, max_bytes=10_000_000, min_free_bytes=0,
                )
            self.assertFalse(any(path.name == "bundle.json" for path in cache_root.rglob("bundle.json")))


if __name__ == "__main__":
    unittest.main()
