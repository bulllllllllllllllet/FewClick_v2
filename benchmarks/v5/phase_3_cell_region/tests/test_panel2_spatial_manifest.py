import unittest

from benchmarks.v5.phase_3_cell_region.src.panel2_spatial_manifest import split_and_buffer


class Panel2SpatialManifestTest(unittest.TestCase):
    def test_cross_split_neighbours_are_excluded(self):
        rows = [
            {"case_id": "case", "patch_name": "a", "footprint_um": [0, 0, 10, 10]},
            {"case_id": "case", "patch_name": "b", "footprint_um": [12, 0, 22, 10]},
        ]
        # Find a seed for which the two adjacent one-unit blocks differ.
        for seed in range(100):
            kept, audit = split_and_buffer(rows, block_um=10, buffer_um=5, seed=seed)
            if audit["excluded_for_cross_split_context"]:
                self.assertEqual(kept, [])
                return
        self.fail("fixture did not create differing block splits")

    def test_same_split_neighbours_are_retained(self):
        rows = [
            {"case_id": "case", "patch_name": "a", "footprint_um": [0, 0, 10, 10]},
            {"case_id": "case", "patch_name": "b", "footprint_um": [1, 0, 11, 10]},
        ]
        kept, audit = split_and_buffer(rows, block_um=100, buffer_um=5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(audit["excluded_for_cross_split_context"], 0)


if __name__ == "__main__":
    unittest.main()
