import unittest

import numpy as np

from benchmarks.v5.tools.audit_crc_candidate_supervision import classify, geometry


class SupervisionAuditTest(unittest.TestCase):
    def test_inclusive_bounds_and_foxp3_combination(self):
        values = np.zeros((3, 6))
        values[:, 0] = [300, 2000, 2001]
        values[:, 4] = 1500
        labels, full, _, _ = classify(values)
        np.testing.assert_array_equal(labels, [2, 2, -1])
        self.assertEqual(full[2], "FOXP3_only_Unresolved")
        self.assertEqual(classify(values, lower_only=True)[0][2], 2)

    def test_competition_and_state_do_not_make_new_identity(self):
        values = np.array([[400, 500, 0, 0, 1600, 600], [0, 0, 0, 0, 0, 600]])
        labels, full, positive, _ = classify(values)
        np.testing.assert_array_equal(labels, [1, -1])
        self.assertEqual(full.tolist(), ["CD8_T_SULF1", "SULF1_only"])
        self.assertEqual(positive[0, :4].sum(), 2)

    def test_geometry_exact_masks_and_wrong_target_rejected(self):
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[12:15, 12:15] = 2
        ids = np.array([1, 2])
        xy = np.array([[3., 3.], [13., 13.]])
        distance, mutual, iou, ratio, _, _ = geometry(mask, mask, ids, xy, ids)
        np.testing.assert_allclose(distance[:, 0], 0)
        self.assertTrue(mutual.all())
        np.testing.assert_allclose(iou, 1)
        np.testing.assert_allclose(ratio, 1)
        with self.assertRaises(AssertionError):
            geometry(mask, mask, ids, xy, ids[::-1])


if __name__ == "__main__":
    unittest.main()
