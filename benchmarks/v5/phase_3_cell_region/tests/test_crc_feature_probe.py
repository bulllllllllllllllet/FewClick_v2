"""Pure invariants of the CRC candidate feature preparation path."""
import unittest
from pathlib import Path

import numpy as np

from benchmarks.v5.tools.prepare_crc_he_feature_probe import broad_class, centers, relocate


class CRCFeatureProbeTest(unittest.TestCase):
    def test_explicit_relocation(self):
        root = Path("/nfs-medical5/hwx/panel2_cell_level_registration")
        old = "/home/hwx/panel2_cell_level_registration/2022-39941E/transforms/x0_y0.json"
        self.assertEqual(relocate(old, root), root / "2022-39941E/transforms/x0_y0.json")
        with self.assertRaises(ValueError):
            relocate("/home/another/data.json", root)

    def test_identity_and_state_separated(self):
        self.assertEqual(broad_class("CD4_T_SULF1"), 0)
        self.assertEqual(broad_class("Treg"), 0)
        self.assertEqual(broad_class("B_cell_SULF1"), 1)
        self.assertEqual(broad_class("CD163_Macrophage"), 2)
        for value in ("Undefined", "SULF1_only", "FOXP3_only_Unresolved_SULF1"):
            self.assertEqual(broad_class(value), -1)

    def test_centers_are_y_x_sorted_but_export_x_y(self):
        mask = np.zeros((8, 8), np.int32)
        mask[1:3, 4:6] = 9
        mask[5:7, 0:2] = 3
        ids, xy = centers(mask)
        np.testing.assert_array_equal(ids, [9, 3])
        np.testing.assert_allclose(xy, [[4.5, 1.5], [.5, 5.5]])

    def test_empty_mask(self):
        ids, xy = centers(np.zeros((4, 4), np.int32))
        self.assertEqual(len(ids), 0)
        self.assertEqual(xy.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
