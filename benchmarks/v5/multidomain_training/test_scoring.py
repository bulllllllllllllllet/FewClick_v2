import unittest

import pyvips

from .evaluate import summarize
from .scoring import split_target_tasks
from .regression import paired


class TissueScoringTests(unittest.TestCase):
    def test_background_ids_are_local_and_cancer_class_two_is_retained(self):
        rows=[dict(dataset='crc',target=2),dict(dataset='ocelot',target=2),
              dict(dataset='ignite',target=1),dict(dataset='puma',target=6),dict(dataset='tiger',target=2)]
        kept,excluded=split_target_tasks(rows)
        self.assertEqual(kept,[rows[1],rows[4]])
        self.assertEqual(excluded,[rows[0],rows[2],rows[3]])

    def test_tissue_queries_without_target_remain_in_false_positive_scores(self):
        absent=dict(dataset='crc',target=0,positive=0,dice=None,bf1_um2=None,all_foreground_dice=0.,false_positive_fraction=.25)
        present=dict(dataset='crc',target=0,positive=4,dice=.5,bf1_um2=.4,all_foreground_dice=.4,false_positive_fraction=.2)
        background=dict(dataset='crc',target=2,positive=8,dice=0.,bf1_um2=0.,all_foreground_dice=.8,false_positive_fraction=.9)
        result=summarize([present,absent,background])
        self.assertEqual(result['episodes'],2)
        self.assertEqual((result['present'],result['absent']),(1,1))
        self.assertEqual(result['dice'],.5)
        self.assertEqual(result['absent_false_positive_fraction'],.25)
        self.assertEqual(result['absent_over_1percent'],1.)
        self.assertEqual(summarize([present,absent,background],include_background_targets=True)['episodes'],3)

    def test_filtering_tasks_never_mutates_their_pixel_counts(self):
        tissue=dict(target_class=0,tp=2,fp=3,fn=1,background_false_positives=3)
        original=tissue.copy();kept,excluded=split_target_tasks([tissue],dataset='crc')
        self.assertIs(kept[0],tissue)
        self.assertEqual(tissue,original)
        self.assertFalse(excluded)

    def test_dataset_is_required_for_legacy_numeric_labels(self):
        with self.assertRaises(ValueError):split_target_tasks([dict(target_class=2)])

    def test_empty_tissue_scope_is_undefined_instead_of_a_perfect_score(self):
        self.assertIsNone(summarize([dict(dataset='crc',target=2)])['dice'])
        result=paired([],'C3')
        self.assertEqual(result['dice']['evaluable_pairs'],0)


if __name__=='__main__':unittest.main()
