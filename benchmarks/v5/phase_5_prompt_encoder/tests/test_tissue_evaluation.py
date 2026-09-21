import unittest
import numpy as np
import torch
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_evaluation import pixel_metrics, task_macro, prototype_logits


class TissueEvaluationTests(unittest.TestCase):
    def test_ignore_and_empty_target(self):
        a = np.array([True, False, True])
        y = np.array([True, True, False])
        r = pixel_metrics(a, y, np.array([True, True, False]))
        self.assertAlmostEqual(r['dice'], 2/3)
        self.assertEqual(r['fp'], 0)
        self.assertIsNone(pixel_metrics(a, np.zeros(3, bool), np.ones(3, bool))['dice'])

    def test_macro_weights_tasks_not_pixels_or_episode_counts(self):
        rows = [dict(task_id='a', dice=1., iou=1.)]*10 + [dict(task_id='b', dice=0., iou=0.)]
        self.assertEqual(task_macro(rows)['task_macro_dice'], .5)

    def test_single_support_single_and_multi_agree(self):
        b = {'fine_tokens': torch.randn(1, 4, 3)}
        for sign in ('positive', 'negative'):
            b[sign+'_tokens'] = torch.randn(1, 1, 3)
            b[sign+'_mask'] = torch.ones(1, 1, dtype=torch.bool)
        torch.testing.assert_close(prototype_logits(b, 'single'), prototype_logits(b, 'multi'))

if __name__ == '__main__': unittest.main()
