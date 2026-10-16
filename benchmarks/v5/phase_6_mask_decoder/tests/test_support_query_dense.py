import unittest
from types import SimpleNamespace
import json
import pyvips
import numpy as np
import pandas as pd
import torch
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v4.phase_6_mask_decoder.src.model import ContextAwareMaskDecoder
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import encode_support, query_features
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import build_pairs, paired_intervals


class SupportQueryTests(unittest.TestCase):
    def test_paired_report_excludes_undefined_scores_without_zero_imputation(self):
        rows = []
        for i, (local_score, remote_score) in enumerate([(None,None),(None,.8),(.6,None),(.5,.7)]):
            for protocol, score in [('local',local_score),('same_slide_present',remote_score)]:
                rows.append(dict(position=i, model='dense', protocol=protocol,
                    query_spatial_group=str(i), dice=.5 if protocol == 'local' else .6,
                    bf1_2=score, unprompted_dice=None))
        report = paired_intervals(rows)['same_slide_present']['dense']
        self.assertEqual(report['paired_episodes'],4)
        self.assertEqual(report['dice']['evaluable_pairs'],4)
        self.assertEqual(report['bf1_2']['evaluable_pairs'],1)
        self.assertAlmostEqual(report['bf1_2']['remote_minus_local'],.2)
        np.testing.assert_allclose(report['bf1_2']['spatial_block_bootstrap_95'],[.2,.2])
        self.assertEqual(report['unprompted_dice']['status'],'no evaluable paired tasks')

    def test_remote_coordinates_do_not_create_query_clicks_but_task_changes(self):
        torch.manual_seed(7)
        model = SimpleNamespace(decoder=ContextAwareMaskDecoder(PromptRegionModel(dropout=0), dropout=0).eval())
        e = {'tokens': torch.randn(1,4,256), 'xy': torch.tensor([[[.1,.1],[.8,.1],[.1,.8],[.8,.8]]]),
             'area': torch.full((1,4),.25), 'active': torch.ones(1,4,dtype=torch.bool),
             'assignment_low': torch.rand(1,4,8,8).softmax(1), 'deep': torch.randn(1,256,8,8),
             'shallow': torch.randn(1,32,16,16), 'semantic': torch.randn(1,12,8,8)}
        p = {'positive_xy': torch.tensor([[[.1,.1]]]), 'negative_xy': torch.tensor([[[.8,.8]]]),
             'positive_mask': torch.ones(1,1,dtype=torch.bool), 'negative_mask': torch.ones(1,1,dtype=torch.bool),
             'prompt_size_id': torch.zeros(1,dtype=torch.long)}
        original = {k:v.clone() for k,v in p.items()}
        with torch.no_grad():
            task = encode_support(model,e,p)
            local = query_features(model,e,task,p,query_id='a',support_id='a',size=(32,32))
            remote = query_features(model,e,task,p,query_id='b',support_id='a',size=(32,32))
            self.assertEqual(int(remote['prompt_maps'].count_nonzero()),0)
            self.assertGreater(int(local['prompt_maps'].count_nonzero()),0)
            self.assertTrue(torch.equal(local['base'],remote['base']))
            changed = dict(p); changed['positive_xy'] = torch.tensor([[[.8,.1]]])
            updated_task = encode_support(model,e,changed)
            updated = query_features(model,e,updated_task,changed,query_id='b',support_id='a',size=(32,32))
            self.assertGreater(float((updated['base']-remote['base']).abs().max()),1e-6)
            contaminated = p | {'target_class': torch.tensor([999]), 'pixel_gt': torch.full((1,32,32),255)}
            repeated = encode_support(model,e,contaminated)
            self.assertTrue(torch.equal(repeated['task_token'],task['task_token']))
        self.assertTrue(all(torch.equal(p[k],original[k]) for k in p))

    def test_pairing_keeps_count_source_and_empty_target_constraints(self):
        class Episodes(SimpleNamespace):
            def __len__(self):
                return len(self.patch_ids)
        ep=Episodes(wsi_ids=np.array(['s','s','s','other','s']),
            target_classes=np.array([0,0,1,0,0]), prompt_sizes=np.array(['point']*5),
            positive_slot_sets=np.array(['[0]','[0]','[0]','[0]','[0,1]']),
            patch_ids=np.array(['q','remote','absent','other','wrong_budget']))
        ids=list(ep.patch_ids)
        spatial=pd.DataFrame({'patch_id':ids,'wsi_id':ep.wsi_ids,'source_key':ep.wsi_ids,'split':['val']*5,
                 'context_x0':[0,2048,4096,0,6144], 'context_y0':[0]*5,
                 'context_x1':[512,2560,4608,512,6656], 'context_y1':[512]*5,
                 'spatial_group_id':ids}).set_index('patch_id')
        patches=pd.DataFrame({'patch_id':ids,'present_classes':[json.dumps(x) for x in [[0,1],[0,1],[1],[0,1],[0,1]]]}).set_index('patch_id')
        refs=[{'identity':{'patch_id':'q','target_class':0,'episode_index':0}}]
        pairs,skipped=build_pairs(ep,refs,spatial,patches)
        self.assertFalse(skipped)
        found={p['protocol']:p for p in pairs}
        self.assertEqual(found['same_slide_present']['support_id'],'remote')
        self.assertEqual(found['same_slide_absent']['query_id'],'absent')
        self.assertEqual(found['cross_slide_present']['support_id'],'other')
        # No valid remote exists after shrinking its context gap; never relax it.
        spatial.loc['remote',['context_x0','context_x1']]=[256,768]
        pairs,skipped=build_pairs(ep,refs,spatial,patches)
        self.assertIn('same_slide_present',[p['protocol'] for p in skipped])


if __name__ == '__main__':
    unittest.main()
