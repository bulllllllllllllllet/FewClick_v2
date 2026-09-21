import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


MODULE = Path(__file__).parents[1] / "tools" / "build_crc_patch_region_alignment.py"
SPEC = importlib.util.spec_from_file_location("crc_patch_region_alignment", MODULE)
ALIGNMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ALIGNMENT)


class CrcPatchRegionAlignmentTest(unittest.TestCase):
    def test_identity_transform_preserves_patch_corners(self):
        actual = ALIGNMENT.corners(10, 20, 5, np.eye(3))
        np.testing.assert_array_equal(actual, [[10, 20], [15, 20], [10, 25], [15, 25]])

    def test_only_strictly_intersecting_grids_are_returned(self):
        inventory = pd.DataFrame({"grid_x": [0, 1, 2], "grid_y": [0, 0, 0], "grid_um": [10, 10, 10]})
        actual = ALIGNMENT.intersecting_regions(inventory, np.array([9.0, 1.0]), np.array([21.0, 9.0]))
        self.assertEqual(actual.grid_x.tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
