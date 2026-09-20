import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "build_crc_tumor_roi_manifest.py"
SPEC = importlib.util.spec_from_file_location("crc_roi_manifest", TOOL); MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class CrcTumorRoiManifestTest(unittest.TestCase):
    def test_builder_preserves_positive_only_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); he = root / "panel_2" / "HE"; he.mkdir(parents=True)
            for case in MODULE.CASES:
                (he / f"{case}.svs").write_bytes(b"fixture")
                (he / f"{case}.geojson").write_text(json.dumps({"type":"FeatureCollection","features":[{"geometry":{"type":"Polygon","coordinates":[[[0,0],[20000,0],[20000,20000],[0,20000],[0,0]]]},"properties":{"classification":{"name":"Tumor"}}}]}))
            spec = MODULE.build_spec(root, 100., 200., 25., .25)
            self.assertFalse(spec["training_ready"]); self.assertTrue(spec["samples"])
            self.assertTrue(all(row["supervision_scope"] == "positive_roi_only" and not row["negative_supervision_available"] for row in spec["samples"]))


if __name__ == "__main__": unittest.main()
