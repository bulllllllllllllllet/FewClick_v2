import importlib.util
from pathlib import Path
import unittest

import numpy as np


TOOL = Path(__file__).resolve().parents[2] / "tools" / "adopt_crc_expert_reviewed_labels.py"
SPEC = importlib.util.spec_from_file_location("crc_label_adoption", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CrcExpertLabelAdoptionTest(unittest.TestCase):
    def test_accepts_only_known_existing_candidates(self):
        mask = MODULE.accepted_identity_mask(np.array([True, True, False, True, True]), np.array([0, 4, 1, -1, 5]))
        np.testing.assert_array_equal(mask, np.array([True, True, False, False, False]))

    def test_rejects_misaligned_arrays(self):
        with self.assertRaises(ValueError):
            MODULE.accepted_identity_mask(np.array([True]), np.array([0, 1]))
