import unittest
import pyvips
import numpy as np
import torch
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.phase_6_mask_decoder.evaluate_crc_resolution import input_rgb,degrade_rgb,select_positions


class ResolutionControlTests(unittest.TestCase):
    def test_rgb_roundtrip_does_not_modify_original_image(self):
        rng=np.random.default_rng(11);rgb=rng.integers(0,256,(512,512,3),dtype=np.uint8)
        np.testing.assert_array_equal(input_rgb(he_to_tensor(rgb)[None]),rgb)

    def test_physical_downsampling_restores_extent_and_removes_high_frequency(self):
        y,x=np.indices((512,512));im=np.repeat((((x+y)%2)*255).astype(np.uint8)[...,None],3,2)
        restored,low=degrade_rgb(im,.504,.504,.777664)
        self.assertEqual(restored.shape,im.shape);self.assertEqual(low.shape,(332,332,3))
        self.assertLess(restored.std(),im.std()/4)
        self.assertLess(abs(low.shape[1]*.777664-512*.504),.777664/2)
        with self.assertRaises(ValueError):degrade_rgb(im,.504,.504,.25)

    def test_selection_balances_classes_and_is_reproducible_without_replacement(self):
        refs=[{'identity':{'target_class':c}} for c in (0,1,2) for _ in range(10)]
        selected=select_positions(refs,2);self.assertEqual(len(selected),6)
        self.assertEqual(len(set(selected)),6);self.assertEqual(selected,select_positions(refs,2))
        for c in (0,1,2):self.assertEqual(sum(refs[i]['identity']['target_class']==c for i in selected),2)


if __name__=='__main__':unittest.main()
