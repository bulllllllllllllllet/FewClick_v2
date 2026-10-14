import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel
from benchmarks.v5.phase_6_mask_decoder.point_boundary_detail import PointBoundaryDetail,sample_features,training_indices


class PointDetailTests(unittest.TestCase):
    def inputs(self):
        return {'deep':torch.randn(1,256,8,8),'shallow':torch.randn(1,32,16,16),
                'semantic':torch.randn(1,12,8,8),'task':torch.randn(1,256),
                'base':torch.rand(1,32,32)*.8+.1,'prompt_maps':torch.rand(1,2,16,16),
                'image':torch.randn(1,3,32,32)}

    def test_pixel_center_sampling(self):
        feature=torch.arange(4.).reshape(1,1,2,2)
        result=sample_features(feature,torch.tensor([[0,3]]),2,2)
        self.assertTrue(torch.equal(result,torch.tensor([[[0.,3.]]])))

    def test_zero_init_and_exact_preservation_outside_selection(self):
        model=PointBoundaryDetail(DenseContextModel().state_dict(),128)
        sample=self.inputs();initial=model(sample)
        self.assertTrue(torch.allclose(initial['pixel_probability'],initial['coarse_probability'],atol=1e-6))
        torch.nn.init.normal_(model.point[-1].weight,std=.1)
        out=model(sample);keep=torch.ones(1,1024,dtype=torch.bool);keep.scatter_(1,out['indices'],False)
        self.assertTrue(torch.equal(out['pixel_probability'].flatten(1)[keep],out['coarse_probability'].flatten(1)[keep]))
        changed=model({**sample,'truth':torch.zeros(1,32,32,dtype=torch.bool),'valid':torch.zeros(1,32,32,dtype=torch.bool)})
        self.assertTrue(torch.equal(changed['indices'],out['indices']))
        self.assertTrue(torch.equal(changed['pixel_probability'],out['pixel_probability']))

    def test_training_sampling_respects_ignore_and_guarantees_both_signs(self):
        p=torch.linspace(0,1,1024).reshape(1,32,32);truth=p>.5;valid=torch.ones_like(truth);valid[:,:,0]=False
        a=training_indices(p,truth,valid,'uniform',17,64)
        b=training_indices(p,truth,valid,'uncertain',17,64)
        self.assertTrue(torch.equal(a[:,:16],b[:,:16]))
        for indices in [a,b]:
            self.assertTrue(valid.flatten()[indices].all())
            targets=truth.flatten()[indices];self.assertTrue(targets.any() and (~targets).any())


if __name__=='__main__':unittest.main()
