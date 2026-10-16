import unittest
import pyvips
import numpy as np
from benchmarks.v5.phase_6_mask_decoder.evaluate_ocelot_external import crop_boxes, support_prompts
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import cells_from_instances
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features


class OcelotExternalTests(unittest.TestCase):
    def test_support_points_follow_labels_and_budget_after_target_switch(self):
        labels=np.ones((512,512),np.uint8);labels[:,256:]=2;labels[:20]=255
        for target in (1,2):
            p=support_prompts(labels,target)
            self.assertEqual(sum(p['positive_mask'][0]),3);self.assertEqual(sum(p['negative_mask'][0]),2)
            self.assertEqual(p['prompt_size_id'],[1])
            for sign,expected in [('positive',target),('negative',3-target)]:
                for x,y in p[sign+'_xy'][0]:self.assertEqual(labels[int(y*512),int(x*512)],expected)

    def test_boxes_exclude_overlap_and_preserve_128_micron_gap(self):
        self.assertFalse(crop_boxes(1024,1024))
        boxes=crop_boxes(1600,1600);self.assertEqual(len(boxes),4)
        for s,q,gap in boxes:
            self.assertEqual((s[2]-s[0],s[3]-s[1]),(512,512))
            self.assertGreaterEqual(gap,128)
            self.assertTrue(s[2]<=q[0] or q[2]<=s[0] or s[3]<=q[1] or q[3]<=s[1])

    def test_detected_geometry_and_unknown_class_keep_74_feature_contract(self):
        inst=np.zeros((100,100),np.int32);inst[10:20,20:30]=1;inst[50:60,60:70]=2
        meta,ids,total=cells_from_instances(inst,max_cells=1)
        self.assertEqual(total,2);self.assertEqual(ids.tolist(),[1]);self.assertEqual(meta[0,3],0)
        np.testing.assert_allclose(meta[0,:3],[.245,.145,np.log1p(100)])
        features=encode_xcell_features(meta,np.ones((1,64),np.float32))
        self.assertEqual(features.shape,(1,74));self.assertEqual(features[0,3],1)
        self.assertEqual(float(features[0,4:10].sum()),0)


if __name__=='__main__':unittest.main()
