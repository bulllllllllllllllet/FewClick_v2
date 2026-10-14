import unittest
from types import SimpleNamespace
import numpy as np
import torch
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel
from benchmarks.v5.phase_6_mask_decoder.train_dense_boundary import select_indices


class Phase2Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Module()
        self.backbone.layer4=torch.nn.Sequential(torch.nn.Conv2d(3,256,1),torch.nn.BatchNorm2d(256))
        self.embedding=torch.nn.Identity()
        self.semantic=torch.nn.Conv2d(256,12,1)

    def extract_backbone_features(self,image):return self.backbone.layer4(image)


class ContinuationTests(unittest.TestCase):
    def test_expansion_keeps_original_prefix_and_unique_patches(self):
        class EpisodeFixture(SimpleNamespace):
            def __len__(self):return len(self.patch_ids)
        ep=EpisodeFixture(patch_ids=np.array([f'p{i//2}' for i in range(240)]),target_classes=np.arange(240)%12)
        dataset=SimpleNamespace(episodes=ep)
        small=select_indices(dataset,24,20261014);large=select_indices(dataset,72,20261014)
        self.assertEqual(large[:24],small)
        self.assertEqual(len(set(ep.patch_ids[large])),72)

    def test_layer4_has_gradient_while_bn_statistics_stay_frozen(self):
        model=DenseContextModel(Phase2Fixture()).train()
        torch.nn.init.normal_(model.head.output.weight,std=.1)
        sample={'image':torch.randn(1,3,8,8),'shallow':torch.randn(1,32,16,16),
                'task':torch.randn(1,256),'base':torch.ones(1,32,32)*.5,'prompt_maps':torch.rand(1,2,16,16)}
        bn=model.phase2.backbone.layer4[1];before=bn.running_mean.clone()
        model(sample)['pixel_logits'].square().mean().backward()
        self.assertFalse(bn.training)
        self.assertTrue(torch.equal(before,bn.running_mean))
        self.assertGreater(float(model.phase2.backbone.layer4[0].weight.grad.abs().sum()),0.)
        self.assertIsNone(model.phase2.semantic.weight.grad)


if __name__=='__main__':unittest.main()
