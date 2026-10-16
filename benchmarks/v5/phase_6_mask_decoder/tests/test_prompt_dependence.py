import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import pyvips
import torch
from benchmarks.v5.phase_6_mask_decoder.evaluate_prompt_dependence import (
    reconstruct_index, select_wrong, zero_task, controls_report)


class PromptDependenceTests(unittest.TestCase):
    def test_wrong_support_changes_class_without_changing_source_or_budget(self):
        ep = SimpleNamespace(patch_ids=['s','s','s','other'],prompt_sizes=['small']*4,
            positive_slot_sets=['[0]','[1]','[0,1]','[0]'],target_classes=[1,2,3,4])
        class Episodes(SimpleNamespace):
            def __len__(self): return len(self.patch_ids)
        ep = Episodes(**vars(ep))
        pair = dict(support_id='s',prompt_size='small',positive_count=1,target_class=1,position=0,source_number=7)
        chosen, skipped = select_wrong(ep,[pair])
        self.assertFalse(skipped); self.assertEqual(chosen[0]['wrong_index'],1)
        chosen, skipped = select_wrong(ep,[pair | {'positive_count':2,'target_class':3}])
        self.assertIsNone(chosen[0]['wrong_index']);self.assertEqual(len(skipped),1)

    def test_zero_control_clears_all_matching_inputs_without_mutation(self):
        original = {k:torch.randn(1,256) for k in ('task_token','q_positive','q_negative')}
        saved = {k:v.clone() for k,v in original.items()}
        result = zero_task(original)
        self.assertTrue(all(v.count_nonzero()==0 for v in result.values()))
        self.assertTrue(all(torch.equal(original[k],saved[k]) for k in original))

    def test_cache_first_use_order_and_unexpected_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'encoded').mkdir()
            for n in range(3): (root/'encoded'/f'{n:05d}.pth').touch()
            pairs=[dict(query_id='q',support_id='s'),dict(query_id='q',support_id='t')]
            index=reconstruct_index(pairs,root)
            self.assertEqual([Path(index[p]).name for p in ('q','s','t')],['00000.pth','00001.pth','00002.pth'])
            (root/'encoded'/'untracked.pth').touch()
            with self.assertRaises(ValueError): reconstruct_index(pairs,root)

    def test_report_compares_wrong_only_to_its_paired_correct_subset(self):
        rows=[]
        for n,condition,dice in [(0,'correct',.8),(1,'correct',.2),(0,'wrong',.3)]:
            r=dict(source_number=n,protocol='same_slide_present',model='dense',condition=condition,
                target_class=1,query_spatial_group=str(n),dice=dice,tp=1,fp=1,fn=1,
                changed_fraction=.5,mask_iou_with_correct=.4,probability_mae=.2)
            r.update({k:None for k in ('bf1_1','bf1_2','bf1_4','boundary_band_iou_2','boundary_band_iou_4','gt_boundary_band_accuracy')})
            rows.append(r)
        result=controls_report(rows)['same_slide_present']['dense']['wrong']
        self.assertEqual(result['correct_dice_on_same_subset'],.8)
        self.assertAlmostEqual(result['dice_delta']['control_minus_correct'],-.5)
        self.assertEqual(result['bf1_2_delta']['evaluable_pairs'],0)


if __name__ == '__main__': unittest.main()
