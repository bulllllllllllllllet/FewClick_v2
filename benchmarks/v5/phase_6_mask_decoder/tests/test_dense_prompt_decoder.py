import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.dense_prompt_decoder import DensePromptDecoder


class DenseTests(unittest.TestCase):
    def inputs(self):
        return {'deep':torch.randn(1,256,8,8),'shallow':torch.randn(1,32,16,16),
                'semantic':torch.randn(1,12,8,8),'task':torch.randn(1,256),
                'base':torch.rand(1,32,32)*.8+.1,'prompt_maps':torch.rand(1,2,16,16)}

    def test_zero_initialization_both_resolutions(self):
        s=self.inputs()
        for high in [False,True]:
            out=DensePromptDecoder(high)(s)
            self.assertTrue(torch.allclose(out['pixel_probability'],s['base'],atol=1e-6))

    def test_prompt_task_changes_trained_output(self):
        head=DensePromptDecoder();torch.nn.init.normal_(head.output.weight,std=.1)
        s=self.inputs();first=head(s)['pixel_probability'];s['task']=s['task']+torch.randn_like(s['task'])*3
        second=head(s)['pixel_probability']
        self.assertGreater(float((first-second).abs().max().detach()),1e-5)

    def test_high_resolution_branch_receives_gradient(self):
        head=DensePromptDecoder(True);torch.nn.init.normal_(head.output.weight,std=.1)
        head(self.inputs())['pixel_logits'].square().mean().backward()
        self.assertGreater(float(head.shallow[0].weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
