import unittest
from benchmarks.v5.phase_5_prompt_encoder.tools.summarize_tissue_learning_curve import summarize


class CurveSummaryTests(unittest.TestCase):
    def test_steps_and_controls_are_separate(self):
        def row(method, step, seed, dice):
            return dict(method=method, step=step, seed=seed, scope='unprompted',
                        task_id='0', dice=dice, iou=dice)
        summary = summarize([row('single', 0, 17, .2), row('single', 100, 17, .8),
                             row('single', 100, 29, .6), row('identity_single', 0, -1, .3)])
        indexed = {(r['method'],r['step']):r for r in summary}
        self.assertAlmostEqual(indexed['single',100]['dice_mean'], .7)
        self.assertEqual(indexed['single',0]['dice_mean'], .2)
        self.assertIsNone(indexed['identity_single',0]['dice_seed_sd'])
