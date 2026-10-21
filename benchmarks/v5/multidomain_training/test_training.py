import unittest
import numpy as np
import torch

from .catalog import interior_labels,split_overlap_audit,footprint
from .episodes import EpisodeCorpus,spread_clicks
from .losses import mask_loss,episode_loss,point_loss
from .model import pixel_prompt_tokens


class GeometryTests(unittest.TestCase):
    def test_interiors_keep_class_zero_and_exclude_unknown_neighbourhood(self):
        labels=np.zeros((40,40),np.uint8);labels[:,20:]=1;labels[15:18,10:13]=255
        inside=interior_labels(labels,[.5,.5])
        self.assertEqual(int(inside[8,8]),0);self.assertEqual(int(inside[20,30]),1)
        self.assertTrue((inside[11:22,6:17]==255).all())
        points=spread_clicks(inside==0,5,np.random.default_rng(1))
        self.assertEqual(len({tuple(p) for p in points}),5)
        for x,y in points:self.assertEqual(int(inside[int(y*40),int(x*40)]),0)

    def test_physical_overlap_rejected_without_patient_grouping(self):
        a=dict(tile_id='a',space_key='same-slide',split='train',x0_um=0,y0_um=0,x1_um=256,y1_um=256)
        b=a|dict(tile_id='b',split='val',x0_um=200,x1_um=456)
        with self.assertRaises(ValueError):split_overlap_audit([a,b])
        self.assertEqual(split_overlap_audit([a,b|dict(x0_um=256,x1_um=512)])['status'],'passed')

    def test_source_origin_is_included_in_physical_footprint(self):
        row=dict(dataset='crc',image_box_json='[4, 4, 1016, 1016]')
        source=dict(extra_json='{"image_window":[1000,2000,1024,1024]}',available_mpp_x=.25,available_mpp_y=.25,slide_id_verified=True,slide_id='s')
        key,box,verified=footprint(row,source)
        self.assertEqual(key,'crc|s');np.testing.assert_allclose(box,[251,501,505,755]);self.assertTrue(verified)

    def test_remote_pairing_distinguishes_slide_and_unverified_ROI(self):
        corpus=EpisodeCorpus.__new__(EpisodeCorpus)
        corpus.boxes=np.array([[0,0,256,256],[200,0,456,256],[512,0,768,256],[0,0,256,256]])
        corpus.keys=np.array(['s','s','s','other'],object);corpus.verified=np.ones(4,bool);corpus.patients=np.array(['p','p','p','q'],object)
        np.testing.assert_array_equal(corpus.candidates(0,np.arange(4),'same'),[2])
        np.testing.assert_array_equal(corpus.candidates(0,np.arange(4),'cross'),[3])
        corpus.verified[:]=False;corpus.keys[1]='another-ROI'
        np.testing.assert_array_equal(corpus.candidates(0,np.arange(4),'cross'),[3])

    def test_quality_arm_keeps_underlying_task_and_clicks_matched(self):
        corpus=EpisodeCorpus.__new__(EpisodeCorpus);corpus.seed=123
        corpus.rows=[dict(tile_id='a',organ='colon',input_mpp_xy_json='[0.5,0.5]',slide_verified=True)]
        corpus.index={'a':0};corpus.prompt_policy='representative'
        corpus.pools={'train':{'crc':{'local':{'colon':{1:[0]}}}}}
        labels=np.zeros((64,64),np.uint8);labels[:,:32]=1;corpus.labels=lambda index:labels
        original=corpus.plan(10,dataset='crc',kind='local',quality_aug=False)
        augmented=corpus.plan(10,dataset='crc',kind='local',quality_aug=True)
        for key in ['support_id','query_id','target','positive_xy','negative_xy','kind']:self.assertEqual(original[key],augmented[key])


class LossTests(unittest.TestCase):
    def test_mixed_empty_and_full_targets_have_finite_correctly_signed_gradients(self):
        valid=torch.ones(1,16,16,dtype=torch.bool);valid[:,:2]=False
        for target in ['mixed','empty','full']:
            truth=torch.zeros_like(valid)
            if target=='mixed':truth[:,:,8:]=True
            elif target=='full':truth[:]=True
            logits=torch.zeros(1,16,16,requires_grad=True)
            loss,_=mask_loss(dict(pixel_logits=logits),truth,valid,[.5,.5]);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(int(logits.grad[~valid].count_nonzero()),0)
            if (truth&valid).any():self.assertTrue((logits.grad[truth&valid]<0).all())
            if (~truth&valid).any():self.assertTrue((logits.grad[~truth&valid]>0).all())

    def test_ignore_hole_does_not_create_a_supervised_boundary(self):
        valid=torch.ones(1,32,32,dtype=torch.bool);valid[:,10:20,10:20]=False
        _,parts=mask_loss(dict(pixel_logits=torch.zeros(1,32,32)),valid.clone(),valid,[.5,.5])
        self.assertEqual(parts['boundary'],0.)

    def test_remote_signed_loss_applies_to_source_only(self):
        logits=torch.zeros(1,16,16,requires_grad=True);source=torch.full((1,16,16),2.,requires_grad=True)
        valid=torch.ones_like(logits,dtype=torch.bool);truth=torch.zeros_like(valid)
        prompts=dict(positive_xy=torch.tensor([[[.25,.25]]]),negative_xy=torch.tensor([[[.75,.75]]]),positive_mask=torch.ones(1,1,dtype=torch.bool),negative_mask=torch.ones(1,1,dtype=torch.bool))
        supervision=dict(truth=truth,valid=valid,support_truth=truth,support_valid=valid,mpp=[.5,.5],support_mpp=[.5,.5])
        output=dict(query=dict(pixel_logits=logits),support=dict(pixel_logits=source))
        actual,_=episode_loss(output,supervision,prompts,False)
        q,_=mask_loss(output['query'],truth,valid,[.5,.5]);s,_=mask_loss(output['support'],truth,valid,[.5,.5])
        expected=q+.25*s+.1*point_loss(source,prompts)
        torch.testing.assert_close(actual,expected)


class PromptTests(unittest.TestCase):
    def test_pixel_assignment_reads_the_clicked_location_and_retains_gradients(self):
        assignment=torch.zeros(1,2,4,4);assignment[:,0,:,:2]=1;assignment[:,1,:,2:]=1;assignment.requires_grad_()
        tokens=torch.tensor([[[2.,0.],[0.,3.]]],requires_grad=True)
        encoded=dict(assignment_low=assignment,tokens=tokens)
        output=pixel_prompt_tokens(encoded,torch.tensor([[[.125,.375],[.875,.375]]]),torch.tensor([[True,False]]))
        torch.testing.assert_close(output,torch.tensor([[[2.,0.],[0.,0.]]]))
        output.sum().backward();self.assertGreater(float(assignment.grad.abs().sum()),0);self.assertGreater(float(tokens.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
