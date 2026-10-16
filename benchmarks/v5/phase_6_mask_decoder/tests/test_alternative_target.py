import unittest
import pyvips
import numpy as np
from benchmarks.v5.phase_6_mask_decoder.score_alternative_target import score_saved, summarize_alternative, completed_pairs


class AlternativeTargetTests(unittest.TestCase):
    def test_rescoring_uses_immediate_run_mask_index_not_inherited_task_number(self):
        pairs=[dict(number=3,source_number=3,wrong_index=None),dict(number=7,source_number=7,wrong_index=91)]
        chosen=completed_pairs({'pairs':pairs},pairs)
        self.assertEqual(chosen[0]['number'],1)
        self.assertEqual(chosen[0]['source_number'],7)

    def make_example(self):
        labels=np.ones((32,32),dtype=np.uint8);labels[:,16:]=2;labels[0]=255
        saved={'truth':labels==1,'valid':labels!=255}
        for model in ('j5','dense'):
            for condition,mask in [('correct',labels==1),('wrong',labels==2),('zero_task',np.zeros_like(labels,dtype=bool))]:
                saved[condition+'_'+model]=mask
        return labels,saved

    def test_prediction_following_new_class_gets_perfect_alternative_score(self):
        labels,saved=self.make_example();rows=score_saved(labels,saved,1,2)
        for r in rows:
            r.update(protocol='same_slide_present',number=0,query_spatial_group='a',wrong_target_class=2)
        report=summarize_alternative(rows)['same_slide_present']['dense']
        self.assertEqual(report['present']['wrong']['dice'],1.)
        self.assertEqual(report['present']['correct']['dice'],0.)
        self.assertEqual(report['present']['wrong']['original_target_dice'],0.)
        self.assertEqual(report['present']['wrong']['new_prompt_minus_original_prompt_dice']['mean'],1.)

    def test_absent_target_uses_false_positive_area_not_empty_dice(self):
        labels,saved=self.make_example();rows=score_saved(labels,saved,1,3)
        wrong=next(r for r in rows if r['condition']=='wrong')
        self.assertFalse(wrong['alternative_present']);self.assertNotIn('dice',wrong)
        self.assertAlmostEqual(wrong['false_positive_area'],.5)

    def test_gt_valid_or_original_class_mismatch_stops_scoring(self):
        labels,saved=self.make_example();saved['valid'][1,1]=False
        with self.assertRaisesRegex(ValueError,'GT differs'):score_saved(labels,saved,1,2)
        labels,saved=self.make_example();labels[2,2]=2
        with self.assertRaisesRegex(ValueError,'GT differs'):score_saved(labels,saved,1,2)


if __name__=='__main__':unittest.main()
