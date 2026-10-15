import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel
from benchmarks.v5.phase_6_mask_decoder.pixel_prompt_decoder import PixelPromptDenseModel, sample_click_features


class PixelPromptTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.state = DenseContextModel().state_dict()
        self.state['head.output.weight'].normal_(std=.1)

    def sample(self):
        return {'deep':torch.randn(1,256,4,4), 'shallow':torch.randn(1,32,8,8),
                'semantic':torch.randn(1,12,4,4), 'task':torch.randn(1,256),
                'base':torch.rand(1,32,32)*.8+.1, 'prompt_maps':torch.rand(1,2,8,8),
                'positive_xy':torch.tensor([[[.2,.3],[.7,.8],[.9,.1]]]),
                'negative_xy':torch.tensor([[[.3,.8],[.8,.3]]]),
                'positive_mask':torch.tensor([[True,True,False]]),
                'negative_mask':torch.tensor([[True,True]])}

    def active_model(self, mode='per_click'):
        model=PixelPromptDenseModel(self.state,mode).eval()
        torch.nn.init.normal_(model.interaction.output.weight,std=.1)
        return model

    def test_click_sampling_respects_pixel_centers_and_padding(self):
        feature=torch.tensor([[[[1.,2.],[3.,4.]]]])
        xy=torch.tensor([[[.25,.25],[.75,.75],[float('nan'),float('nan')]]])
        values=sample_click_features(feature,xy,torch.tensor([[True,True,False]]))
        torch.testing.assert_close(values,torch.tensor([[[1.],[4.],[0.]]]),rtol=0,atol=0)

    def test_all_arms_exactly_reproduce_coarse_at_initialization(self):
        sample=self.sample();coarse=DenseContextModel();coarse.load_state_dict(self.state)
        expected=coarse(sample)['pixel_probability']
        for mode in ['control','pooled','per_click']:
            self.assertTrue(torch.equal(expected,PixelPromptDenseModel(self.state,mode)(sample)['pixel_probability']))

    def test_signed_sets_are_permutation_invariant_and_ignore_padding(self):
        sample=self.sample();model=self.active_model()
        expected=model(sample)['pixel_logits']
        altered={**sample,'positive_xy':sample['positive_xy'][:,[1,0,2]],
                 'positive_mask':sample['positive_mask'][:,[1,0,2]]}
        altered['positive_xy']=altered['positive_xy'].clone()
        altered['positive_xy'][0,2]=float('nan')
        torch.testing.assert_close(expected,model(altered)['pixel_logits'],rtol=0,atol=2e-6)

    def test_gt_is_not_an_inference_input(self):
        sample=self.sample();model=self.active_model()
        expected=model(sample)['pixel_logits']
        altered={**sample,'truth':torch.ones(1,32,32,dtype=torch.bool),
                 'valid':torch.zeros(1,32,32,dtype=torch.bool),'target_class':torch.tensor([11])}
        self.assertTrue(torch.equal(expected,model(altered)['pixel_logits']))

    def test_empty_sets_are_finite(self):
        for mode in ['pooled','per_click']:
            model=self.active_model(mode);sample=self.sample()
            for name in ['positive','negative']:
                sample[name+'_mask']=torch.zeros_like(sample[name+'_mask'])
            self.assertTrue(torch.isfinite(model(sample)['pixel_logits']).all())
            for name in ['positive','negative']:
                sample[name+'_xy']=sample[name+'_xy'][:,:0]
                sample[name+'_mask']=sample[name+'_mask'][:,:0]
            self.assertTrue(torch.isfinite(model(sample)['pixel_logits']).all())

    def test_pooled_control_shares_parameters_but_loses_individual_information(self):
        point=self.active_model();pooled=self.active_model('pooled');pooled.load_state_dict(point.state_dict())
        self.assertEqual(sum(p.numel() for p in point.parameters()),sum(p.numel() for p in pooled.parameters()))
        sample=self.sample()
        self.assertGreater(float((point(sample)['pixel_logits']-pooled(sample)['pixel_logits']).abs().max()),1e-5)
        for name in ['positive','negative']:
            sample[name+'_mask']=torch.zeros_like(sample[name+'_mask']);sample[name+'_mask'][0,0]=True
        torch.testing.assert_close(point(sample)['pixel_logits'],pooled(sample)['pixel_logits'],rtol=0,atol=2e-6)

    def test_visual_and_attention_paths_receive_gradients(self):
        sample=self.sample();model=self.active_model()
        model(sample)['pixel_logits'].square().mean().backward()
        for name in ['deep.weight','shallow.weight','query.weight','key.weight','value.weight','position.weight','sign.weight']:
            grad=dict(model.interaction.named_parameters())[name].grad
            self.assertIsNotNone(grad,name);self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0,name)
        self.assertGreater(float(model.coarse.head.output.weight.grad.abs().sum()),0)

    def test_valid_click_content_and_sign_affect_output(self):
        sample=self.sample();model=self.active_model();expected=model(sample)['pixel_logits']
        moved={**sample,'positive_xy':sample['positive_xy'].clone()};moved['positive_xy'][0,0]=torch.tensor([.85,.15])
        self.assertGreater(float((expected-model(moved)['pixel_logits']).abs().max()),1e-5)
        swapped={**sample,'positive_xy':sample['negative_xy'],'negative_xy':sample['positive_xy'],
                 'positive_mask':sample['negative_mask'],'negative_mask':sample['positive_mask']}
        self.assertGreater(float((expected-model(swapped)['pixel_logits']).abs().max()),1e-5)


if __name__=='__main__':
    unittest.main()
