import unittest
import numpy as np
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_tasks import TissueTask


class TissueTaskTests(unittest.TestCase):
    def test_unknown_is_not_negative_and_composite_task_is_allowed(self):
        task = TissueTask('epithelium', (0, 4), (1, 5))
        target, valid = task.masks(np.array([0, 4, 1, 99, 255]))
        np.testing.assert_array_equal(target, [True, True, False, False, False])
        np.testing.assert_array_equal(valid, [True, True, True, False, False])

    def test_conflicting_classes_fail(self):
        with self.assertRaises(ValueError): TissueTask('bad', (0,), (0,))

if __name__ == '__main__': unittest.main()
