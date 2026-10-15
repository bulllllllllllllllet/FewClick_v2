import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.visualize_test_boundary import boundary_zoom_box


class BoundaryVisualizationTests(unittest.TestCase):
    def test_zoom_is_deterministic_and_within_image(self):
        truth = torch.zeros(1, 32, 32, dtype=torch.bool)
        truth[:, 12:20, 12:20] = True
        valid = torch.ones_like(truth)
        box = boundary_zoom_box(truth, valid, size=8, stride=4)
        self.assertEqual(box, boundary_zoom_box(truth, valid, size=8, stride=4))
        x, y, w, h = box
        self.assertTrue(0 <= x <= 32-w and 0 <= y <= 32-h)
        self.assertGreater(int(truth[0, y:y+h, x:x+w].sum()), 0)
        self.assertLess(int(truth[0, y:y+h, x:x+w].sum()), w*h)


if __name__ == '__main__':
    unittest.main()
