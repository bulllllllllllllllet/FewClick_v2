import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[2]/"tools"/"build_crc_pooled_identity_manifest.py"
SPEC=importlib.util.spec_from_file_location("pooled_manifest",PATH); MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)

class PooledManifestTest(unittest.TestCase):
    def test_each_case_occurs_in_train_and_val_without_duplicate_rows(self):
        rows=[{"sample_id":f"{case}_{i}","case_id":case,"split":"test"} for case in ("a","b","c") for i in range(4)]
        result=MODULE.pooled_rows(rows,1)
        self.assertEqual(len({r["sample_id"] for r in result}),12)
        for case in ("a","b","c"):
            self.assertEqual(sum(r["case_id"]==case and r["split"]=="train" for r in result),3)
            self.assertEqual(sum(r["case_id"]==case and r["split"]=="val" for r in result),1)
