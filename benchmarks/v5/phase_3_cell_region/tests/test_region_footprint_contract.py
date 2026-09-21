import unittest

import numpy as np
import torch

from benchmarks.v5.phase_3_cell_region.src.region_footprint_contract import (
    patch_pixel_grid_indices, project_assignment_to_grid, project_observed_targets,
)


class RegionFootprintContractTest(unittest.TestCase):
    def test_identity_geometry_and_soft_mass_projection(self):
        grid = patch_pixel_grid_indices(2, 2, (0, 0), np.eye(3), 1., 1.)
        np.testing.assert_array_equal(grid, [[[0, 0], [1, 0]], [[0, 1], [1, 1]]])
        assignment = torch.tensor([[[1., 0.], [0., 0.]], [[0., .5], [.5, 1.]]])
        result = project_assignment_to_grid(assignment, grid)
        self.assertEqual(result["token_grid_mass"].shape, (2, 4))
        self.assertTrue(torch.allclose(result["token_grid_fraction"].sum(-1), torch.ones(2)))

    def test_low_resolution_assignment_uses_source_pixel_stride(self):
        grid = patch_pixel_grid_indices(1, 2, (10, 0), np.eye(3), 1., 10., source_pixel_stride=10.)
        np.testing.assert_array_equal(grid, [[[1, 0], [2, 0]]])

    def test_high_overlap_gate_does_not_copy_targets_to_all_tokens(self):
        assignment = torch.tensor([[[1., 1.]], [[0., 0.]]])
        projection = project_assignment_to_grid(assignment, np.array([[[0, 0], [1, 0]]]))
        out = project_observed_targets(projection, {(0, 0): ([2, 1, 0, 0, 0], 3, .9)}, min_token_grid_fraction=.75)
        self.assertFalse(out["immune_target_valid"].any())
        out = project_observed_targets(projection, {(0, 0): ([2, 1, 0, 0, 0], 3, .9)}, min_token_grid_fraction=.5)
        self.assertTrue(out["immune_target_valid"][0])
        self.assertEqual(int(out["immune_counts"][0, 0]), 2)


if __name__ == "__main__":
    unittest.main()
