import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[2]/"tools"/"build_crc_identity_phase3_manifest.py"
spec=importlib.util.spec_from_file_location("crc_phase3_manifest",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

class CrcIdentityContractTest(unittest.TestCase):
    def test_cells_keep_only_expert_verified_labels(self):
        raw={"he_cell_ids":np.array([1,2]),"he_mask":np.array([[1,1],[2,0]],np.int32),"he_xy":np.array([[.5,.5],[0,1]]),"reg_features":np.ones((2,64),np.float32),"identity_verified":np.array([True,False]),"candidate_class":np.array([3,1])}
        cells,valid,allowed=module.cells_from_cache(raw)
        self.assertEqual(cells.shape,(2,74)); self.assertTrue(valid[0]); self.assertFalse(valid[1]); self.assertTrue(allowed[0,3]); self.assertFalse(allowed[1].any()); self.assertEqual(cells[0,3],1.)
