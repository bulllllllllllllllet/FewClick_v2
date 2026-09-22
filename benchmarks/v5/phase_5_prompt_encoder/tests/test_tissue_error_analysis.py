import unittest
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_error_analysis import aggregate_task_errors, paired_delta


class TissueErrorAnalysisTests(unittest.TestCase):
    def test_counts_define_rates(self):
        result = aggregate_task_errors([dict(task_id='0',tp=2,fp=3,fn=2,valid_pixels=10)])['0']
        self.assertAlmostEqual(result['dice'], 4/9)
        self.assertAlmostEqual(result['recall'], .5)
        self.assertAlmostEqual(result['false_positive_rate'], .5)

    def test_delta_matches_task_not_row_order(self):
        local = aggregate_task_errors([dict(task_id='1',tp=4,fp=0,fn=0,valid_pixels=10)])
        cross = aggregate_task_errors([dict(task_id='1',tp=2,fp=0,fn=2,valid_pixels=10)])
        self.assertLess(paired_delta(local,cross)['1']['dice_delta'], 0)
