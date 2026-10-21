"""Scientific comparison contracts for prompt/cell regression diagnosis."""
import unittest

import pyvips
import numpy as np
import torch

from .diagnose_regression import make_interior_prompts, paired_delta, point_audit, unknown_cells


class DiagnosisTests(unittest.TestCase):
    def test_geometric_center_of_ring_is_not_a_positive_pixel_click(self):
        target=np.ones((20,20),bool);target[7:13,7:13]=False
        prompts=dict(positive_xy=torch.tensor([[[.5,.5]]]),positive_mask=torch.tensor([[True]]),
                     negative_xy=torch.tensor([[[.1,.1]]]),negative_mask=torch.tensor([[True]]))
        result=point_audit(prompts,target,np.ones_like(target))
        self.assertEqual(result['positive']['incorrect'],1)
        self.assertEqual(result['negative']['incorrect'],1)

    def test_actual_clicks_keep_budget_size_condition_and_avoid_ignore_interfaces(self):
        labels=np.ones((64,64),np.uint8);labels[8:50,8:32]=0;labels[:4]=255
        historical=dict(prompt_size_id=torch.tensor([2]),positive_mask=torch.tensor([[True,True,True,False]]),
                        negative_mask=torch.tensor([[True,True,False]]))
        prompts,scope=make_interior_prompts(labels,0,[.5,.5],historical,19,'cpu')
        self.assertEqual(scope['status'],'available')
        self.assertEqual(prompts['prompt_size_id'].item(),2)
        self.assertEqual(prompts['positive_mask'].sum().item(),3)
        self.assertEqual(prompts['negative_mask'].sum().item(),2)
        checks=point_audit(prompts,labels==0,labels!=255)
        for sign in ['positive','negative']:
            self.assertEqual(checks[sign]['incorrect'],0)
            self.assertEqual(checks[sign]['ignored'],0)

    def test_thin_target_reports_infeasible_without_moving_or_substituting_it(self):
        labels=np.ones((40,40),np.uint8);labels[:,20]=0
        historical=dict(prompt_size_id=torch.tensor([1]),positive_mask=torch.tensor([[True]]),negative_mask=torch.tensor([[True]]))
        prompts,scope=make_interior_prompts(labels,0,[.5,.5],historical,1,'cpu')
        self.assertIsNone(prompts)
        self.assertEqual(scope,dict(status='infeasible',sign='positive',available=0,requested=1,margin_um=2.))

    def test_unknown_identity_keeps_geometry_reg_features_and_density(self):
        cells=torch.arange(2*74,dtype=torch.float32).reshape(1,2,74)
        original=dict(cells=cells.clone(),cell_valid=torch.ones((1,2),dtype=torch.bool),total_cell_count=torch.tensor([90.]))
        changed=unknown_cells(original)
        self.assertTrue(torch.equal(changed['cells'][...,:3],cells[...,:3]))
        self.assertTrue(torch.equal(changed['cells'][...,10:],cells[...,10:]))
        self.assertTrue(torch.equal(original['cells'],cells))
        self.assertTrue(torch.equal(changed['cells'][...,3:10],torch.tensor([1.,0,0,0,0,0,0]).expand(1,2,7)))
        self.assertIs(changed['total_cell_count'],original['total_cell_count'])

    def test_paired_statistics_require_same_query_and_defined_metric(self):
        reference=[dict(position=0,dice=.8),dict(position=1,dice=None)]
        candidate=[dict(position=0,dice=.9,spatial_group='block'),dict(position=1,dice=.9,spatial_group='block'),
                   dict(position=2,dice=0.,spatial_group='other')]
        result=paired_delta(candidate,reference,'dice')
        self.assertEqual(result['pairs'],1)
        self.assertEqual(result['spatial_blocks'],1)
        self.assertAlmostEqual(result['candidate_minus_reference'],.1)
        self.assertTrue(np.allclose(result['bootstrap95'],[.1,.1]))


if __name__=='__main__':unittest.main()
