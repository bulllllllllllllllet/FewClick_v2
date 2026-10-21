import unittest
import numpy as np

from .click_policy import representative_clicks


class ClickPolicyTests(unittest.TestCase):
    def test_main_tissue_wins_over_many_tiny_islands_and_clicks_spread(self):
        labels=np.ones((256,256),np.uint8);labels[40:200,40:170]=0
        for y in [8,220]:
            for x in range(8,240,20):labels[y:y+12,x:x+12]=0
        clicks,audit=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(3),dataset='crc')
        pixels=(np.array(clicks['positive_xy'])*256).astype(int)
        self.assertTrue(((pixels[:,0]>=40)&(pixels[:,0]<170)&(pixels[:,1]>=40)&(pixels[:,1]<200)).all())
        self.assertGreater(min(audit['positive']['clearance_um']),10)
        self.assertGreater(audit['positive']['minimum_pair_distance_um'],10)

    def test_tissue_negatives_precede_white_background_and_ignore(self):
        labels=np.full((160,160),2,np.uint8);labels[20:110,20:75]=0;labels[20:140,90:140]=1;labels[:5]=255
        clicks,audit=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(2),dataset='crc')
        for x,y in clicks['negative_xy']:self.assertEqual(labels[int(y*160),int(x*160)],1)
        self.assertFalse(audit['background_negative_required'])

    def test_background_negative_allowed_when_it_is_the_only_alternative(self):
        labels=np.full((80,80),2,np.uint8);labels[15:60,15:60]=0
        clicks,audit=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(7),dataset='crc')
        self.assertTrue(audit['background_negative_required'])
        self.assertEqual(audit['negative']['selected_labels'],[2,2])

    def test_ocelot_class_two_is_the_target_and_negative_one_is_tissue(self):
        labels=np.ones((100,100),np.uint8);labels[20:80,20:60]=2
        clicks,audit=representative_clicks(labels,2,[.5,.5],3,2,np.random.default_rng(9),dataset='ocelot')
        self.assertEqual(audit['positive']['selected_labels'],[2,2,2]);self.assertFalse(audit['background_negative_required'])

    def test_negative_organization_classes_are_diverse(self):
        labels=np.zeros((160,160),np.uint8);labels[:80,80:]=1;labels[80:,80:]=3
        _,audit=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(1),dataset='crc')
        self.assertEqual(set(audit['negative']['selected_labels']),{1,3})

    def test_seed_is_reproducible(self):
        labels=np.ones((120,120),np.uint8);labels[15:105,15:65]=0
        a=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(10),dataset='crc')
        b=representative_clicks(labels,0,[.5,.5],3,2,np.random.default_rng(10),dataset='crc')
        self.assertEqual(a,b)

    def test_no_negative_space_raises_without_changing_target_or_budget(self):
        with self.assertRaises(ValueError):representative_clicks(np.zeros((40,40),np.uint8),0,[.5,.5],3,2,np.random.default_rng(1),dataset='crc')


if __name__=='__main__':unittest.main()
