"""Scientific controls for hard click-to-region selection."""
import unittest

import pyvips
import torch

from .check_region_mapping import hard_prompt_tokens, membership
from .model import pixel_prompt_tokens
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import remap_prompt_tokens


class RegionMappingTests(unittest.TestCase):
    def test_nonconvex_membership_can_disagree_with_nearest_centroid(self):
        # Region0 occupies the click while its geometric center is far away.
        encoded=dict(assignment_low=torch.tensor([[[[.9,.1]],[[.1,.9]]]]),
                     tokens=torch.tensor([[[2.,4.],[20.,40.]]]),xy=torch.tensor([[[.9,.5],[.3,.5]]]))
        xy=torch.tensor([[[.25,.5]]]);valid=torch.tensor([[True]])
        nearest=remap_prompt_tokens(encoded['tokens'],encoded['xy'],xy,valid)[0]
        hard=hard_prompt_tokens(encoded,xy,valid);soft=pixel_prompt_tokens(encoded,xy,valid)
        self.assertTrue(torch.equal(nearest,encoded['tokens'][:,1:2]))
        self.assertTrue(torch.equal(hard,encoded['tokens'][:,0:1]))
        self.assertTrue(torch.allclose(soft,torch.tensor([[[3.8,7.6]]])))

    def test_single_region_selection_masks_padding_and_preserves_batch_identity(self):
        encoded=dict(assignment_low=torch.tensor([[[[.1]],[[.9]]],[[[.8]],[[.2]]]]),
                     tokens=torch.tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]]))
        xy=torch.tensor([[[.5,.5],[.5,.5]],[[.5,.5],[.5,.5]]]);valid=torch.tensor([[True,False],[True,True]])
        result=hard_prompt_tokens(encoded,xy,valid)
        self.assertTrue(torch.equal(result,torch.tensor([[[3.,4.],[0.,0.]],[[5.,6.],[5.,6.]]])))
        self.assertTrue(torch.allclose(membership(encoded,xy).sum(1),torch.ones(2,2)))

    def test_no_click_returns_no_tokens(self):
        encoded=dict(assignment_low=torch.ones(1,1,2,2),tokens=torch.ones(1,1,4))
        self.assertEqual(hard_prompt_tokens(encoded,torch.empty(1,0,2),torch.empty(1,0,dtype=torch.bool)).shape,(1,0,4))


if __name__=='__main__':unittest.main()
