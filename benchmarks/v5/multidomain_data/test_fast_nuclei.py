import unittest
import numpy as np

from .fast_nuclei import CachedCenterCropper
from .evaluate_nuclei_points import match_points
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import attended_crop


class CropTests(unittest.TestCase):
    def test_crop_pixels_match_original_including_edge_padding(self):
        rng=np.random.default_rng(42);rgb=rng.integers(0,256,(256,256,3),dtype=np.uint8)
        inst=np.zeros((256,256),np.int32)
        inst[1:7,2:8]=1;inst[250:256,251:256]=2;inst[110:125,130:145]=3;inst[112,132]=0
        cached=CachedCenterCropper()
        for label in [1,3,2,1]:np.testing.assert_array_equal(cached(rgb,inst,label),attended_crop(rgb,inst,label))
        changed=inst.copy();changed[20:30,30:40]=4
        np.testing.assert_array_equal(cached(rgb,changed,4),attended_crop(rgb,changed,4))


class PointMatchingTests(unittest.TestCase):
    def test_valid_cardinality_takes_priority_over_shorter_invalid_assignment(self):
        # Nearest-total-distance assignment can sacrifice a valid second match.
        pred=np.array([[0.,0.],[1.,0.]])
        gt=np.array([[0.,0.],[-.5,0.]])
        self.assertEqual(match_points(pred,gt,1.),dict(tp=2,fp=0,fn=0))

    def test_empty_and_unmatched_points(self):
        self.assertEqual(match_points(np.empty((0,2)),np.zeros((2,2)),3),dict(tp=0,fp=0,fn=2))
        self.assertEqual(match_points(np.array([[0.,0.],[10.,0.]]),np.array([[.1,0.]]),1),dict(tp=1,fp=1,fn=0))


if __name__=='__main__':unittest.main()
