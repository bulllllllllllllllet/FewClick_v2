import unittest
import torch
from benchmarks.v5.phase_6_mask_decoder.segnext_adapter import official_points, rgb_from_normalized
from benchmarks.v5.phase_6_mask_decoder.segnext_vendor.ops import DistMaps


class SegNextInputTests(unittest.TestCase):
    def sample(self):
        return {'positive_xy':torch.tensor([[[11.2/48,5.2/32],[float('nan'),float('nan')]]]),
                'negative_xy':torch.tensor([[[30.2/48,20.2/32]]]),
                'positive_mask':torch.tensor([[True,False]]),'negative_mask':torch.tensor([[True]])}

    def test_inverse_imagenet_normalization(self):
        rgb=torch.rand(1,3,16,16)
        mean=torch.tensor([.485,.456,.406])[None,:,None,None]
        std=torch.tensor([.229,.224,.225])[None,:,None,None]
        torch.testing.assert_close(rgb_from_normalized((rgb-mean)/std),rgb,atol=1e-7,rtol=0)

    def test_points_match_pixel_scoring_convention_and_signed_padding(self):
        points=official_points(self.sample(),32,48)
        torch.testing.assert_close(points[0,0,:2],torch.tensor([5.,11.]),atol=0,rtol=0)
        torch.testing.assert_close(points[0,2,:2],torch.tensor([20.,30.]),atol=0,rtol=0)
        self.assertTrue((points[0,1]==-1).all());self.assertTrue((points[0,3]==-1).all())

    def test_official_disk_maps_place_both_signs_on_the_correct_pixels(self):
        points=official_points(self.sample(),32,48)
        maps=DistMaps(norm_radius=1,use_disks=True)([1,1,32,48],points)
        self.assertEqual(float(maps[0,0,5,11]),1)
        self.assertEqual(float(maps[0,1,20,30]),1)
        self.assertEqual(float(maps[0,1,5,11]),0)
        self.assertEqual(float(maps[0,0,20,30]),0)

    def test_conversion_is_repeatable_without_mutating_cached_coordinates(self):
        sample=self.sample();old=sample['positive_xy'].clone()
        first=official_points(sample,32,48);first.add_(500)
        second=official_points(sample,32,48)
        torch.testing.assert_close(sample['positive_xy'],old,equal_nan=True)
        self.assertEqual(float(second[0,0,0]),5)



if __name__=='__main__':unittest.main()
