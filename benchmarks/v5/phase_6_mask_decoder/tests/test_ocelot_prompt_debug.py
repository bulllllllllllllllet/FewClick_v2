from types import SimpleNamespace
import unittest
import pyvips
import numpy as np
import torch
from benchmarks.v5.phase_6_mask_decoder.debug_ocelot_prompts import spread_points,make_variants,pixel_task


class PromptDebugTests(unittest.TestCase):
    def test_spread_preserves_labels_interior_and_spacing(self):
        mask=np.zeros((512,512),bool);mask[20:490,30:480]=True
        xy=np.array(spread_points(mask,3))*512
        self.assertEqual(torch.tensor(spread_points(mask,3)).dtype,torch.float32)
        for x,y in xy:self.assertTrue(mask[int(y),int(x)])
        for i in range(3):
            for j in range(i):self.assertGreaterEqual(np.linalg.norm(xy[i]-xy[j]),64)
        with self.assertRaises(ValueError):spread_points(np.zeros((20,20),bool),2)

    def test_negative_only_keeps_positive_points_and_budget(self):
        labels=np.ones((512,512),np.uint8);labels[:256]=2
        original={'positive_xy':[[[.1,.1],[.12,.12],[.14,.14]]],'negative_xy':[[[.1,.9],[.13,.9]]],
            'positive_mask':[[True]*3],'negative_mask':[[True]*2],'prompt_size_id':[1]}
        variants,_=make_variants(original,labels,np.full((512,512,3),255,np.uint8))
        self.assertEqual(variants['negative_spread']['positive_xy'],original['positive_xy'])
        self.assertEqual(len(variants['negative_spread']['negative_xy'][0]),2)
        self.assertNotIn('tissue_negative',variants)  # All white cannot be silently relabeled tissue.

    def test_pixel_mapping_uses_assignment_at_click_not_nearest_centroid(self):
        tokens=torch.tensor([[[10.,20.],[30.,40.]]]);assignment=torch.zeros(1,2,2,2)
        assignment[:,0,:,:1]=1;assignment[:,1,:,1:]=1
        encoded={'tokens':tokens,'assignment_low':assignment,'xy':torch.tensor([[[.9,.5],[.1,.5]]])}
        prompts={'positive_xy':torch.tensor([[[.1,.5]]]),'negative_xy':torch.tensor([[[.9,.5]]]),
            'positive_mask':torch.tensor([[True]]),'negative_mask':torch.tensor([[True]]),'prompt_size_id':torch.tensor([1])}
        model=SimpleNamespace(decoder=SimpleNamespace(prompt_model=SimpleNamespace(encode_prompt_task=lambda x:x)))
        task,indices=pixel_task(model,encoded,prompts)
        self.assertEqual(indices,{'positive':[[0]],'negative':[[1]]})
        self.assertTrue(torch.equal(task['positive_tokens'],tokens[:,:1]))
        self.assertTrue(torch.equal(task['negative_tokens'],tokens[:,1:]))
        self.assertTrue(torch.equal(prompts['positive_xy'],torch.tensor([[[.1,.5]]])))


if __name__=='__main__':unittest.main()
