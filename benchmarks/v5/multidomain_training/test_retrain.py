"""Controls for reporting old clicks without changing primary task selection."""
import copy
import unittest

import pyvips
from .train import validate_legacy_plans


class RetrainCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(number=1,support_id='s',query_id='q',dataset='crc',target=0,kind='same_present',protocol='same_wsi',
                      split='val',support_quality='original',query_quality='original',quality_condition='original',pair_id=2,
                      positive_xy=[[.1,.2]],negative_xy=[[.8,.7]])
        self.new=self.old|dict(positive_xy=[[.3,.4]],negative_xy=[[.6,.5]],prompt_policy='representative_1021_214456')

    def test_same_task_changed_coordinates_are_a_valid_control_and_not_mutated(self):
        before=copy.deepcopy((self.new,self.old));validate_legacy_plans([self.new],[self.old])
        self.assertEqual((self.new,self.old),before)

    def test_different_query_target_quality_or_task_count_is_rejected(self):
        for key,value in [('query_id','other'),('target',1),('query_quality','0.8')]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_legacy_plans([self.new],[self.old|{key:value}])
        with self.assertRaises(ValueError):validate_legacy_plans([self.new],[])

    def test_point_budget_and_protocol_are_not_silently_reinterpreted(self):
        with self.assertRaises(ValueError):validate_legacy_plans([self.new],[self.old|dict(positive_xy=[[.1,.2],[.2,.3]])])
        with self.assertRaises(ValueError):validate_legacy_plans([self.new],[self.new])


if __name__=='__main__':unittest.main()
