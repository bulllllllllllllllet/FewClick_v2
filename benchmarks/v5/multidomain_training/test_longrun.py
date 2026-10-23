import unittest
from types import SimpleNamespace

import pyvips

from .longrun import learning_rates,extend_validation,Coverage


class ScheduleTests(unittest.TestCase):
    def test_constant_reproduces_original_learning_rates(self):
        for step in [0,199,200,1999]:self.assertEqual(learning_rates(step,2000,200),dict(heads=1e-4,backbone=1e-5))

    def test_head_and_unfrozen_backbone_have_separate_ramps_and_cosine_floor(self):
        start=learning_rates(0,20000,2000,schedule='warmup_cosine')
        self.assertAlmostEqual(start['heads'],1e-4/200);self.assertEqual(start['backbone'],0.)
        self.assertEqual(learning_rates(1999,20000,2000,schedule='warmup_cosine')['backbone'],0.)
        joint=learning_rates(2000,20000,2000,schedule='warmup_cosine')
        self.assertAlmostEqual(joint['heads'],1e-4);self.assertAlmostEqual(joint['backbone'],1e-5/200)
        end=learning_rates(19999,20000,2000,schedule='warmup_cosine')
        self.assertAlmostEqual(end['heads'],1e-5);self.assertAlmostEqual(end['backbone'],1e-6)
        for step in range(0,20000,137):
            r=learning_rates(step,20000,2000,schedule='warmup_cosine');self.assertTrue(0<r['heads']<=1e-4);self.assertTrue(0<=r['backbone']<=1e-5)

    def test_invalid_schedule_is_rejected(self):
        for kwargs in [dict(step=20,total=20,joint_start=2),dict(step=0,total=20,joint_start=21),dict(step=0,total=20,joint_start=2,ramp=0)]:
            with self.assertRaises(ValueError):learning_rates(**kwargs)


class ValidationAndCoverageTests(unittest.TestCase):
    def test_original_policy_default_and_expansion_cannot_reintroduce_excluded_inputs(self):
        from .evaluate import validation_plans
        class Corpus:
            training_policy=dict(datasets=['crc','tiger','ignite'],qualities=['original'])
            def plan(self,number,**kwargs):
                return dict(number=number,dataset=kwargs['dataset'],kind=kwargs['kind'],split=kwargs['split'],support_id=f's{number}',query_id=f'q{number}',positive_xy=[[.1,.2]],negative_xy=[[.7,.8]],target=1,support_quality='original',query_quality='original')
        corpus=Corpus();anchors=validation_plans(corpus)
        self.assertEqual(len(anchors),60)
        expanded=extend_validation(corpus,anchors,extra_per_stratum=2)
        self.assertEqual(len(expanded),90);self.assertEqual(expanded[:60],anchors)
        for plan in expanded:
            self.assertIn(plan['dataset'],corpus.training_policy['datasets'])
            self.assertEqual((plan['support_quality'],plan['query_quality']),('original','original'))
        for invalid in [anchors[0]|dict(dataset='ocelot'),anchors[0]|dict(query_quality='0.8')]:
            with self.assertRaises(ValueError):extend_validation(corpus,[invalid])

    def test_extended_tasks_preserve_the_anchor_and_matched_quality_pairs(self):
        class Corpus:
            def plan(self,number,**kwargs):return dict(number=number,dataset=kwargs['dataset'],kind=kwargs['kind'],split=kwargs['split'],support_id=f's{number}',query_id=f'q{number}',positive_xy=[[.1,.2]],negative_xy=[[.7,.8]],target=2,support_quality='original',query_quality='original')
        anchors=[dict(number=1,unchanged=True)]
        result=extend_validation(Corpus(),anchors)
        self.assertEqual(result[:1],anchors);self.assertEqual(len(result),801)
        for i in range(1,len(result),2):
            a,b=result[i:i+2]
            for key in ['number','support_id','query_id','positive_xy','negative_xy','target','pair_id']:self.assertEqual(a[key],b[key])
            self.assertEqual(a['quality_condition'],'original');self.assertEqual(b['quality_condition'],'degraded')

    def test_coverage_counts_unique_fields_separately_from_quality_versions(self):
        corpus=SimpleNamespace(rows=[dict(tile_id='s',dataset='crc',split='train'),dict(tile_id='q',dataset='crc',split='train'),dict(tile_id='v',dataset='crc',split='val')],eligible=[{0},set(),{0}],index={'s':0,'q':1,'v':2})
        coverage=Coverage(corpus);plan=dict(dataset='crc',split='train',support_id='s',query_id='q',query_quality='original')
        coverage.observe(plan);coverage.observe(plan);coverage.observe(plan|dict(query_quality='0.8'))
        report=coverage.summary();self.assertEqual(report['query_pool_tiles'],2);self.assertEqual(report['seen_query_tiles'],1)
        self.assertEqual(report['by_dataset']['crc']['seen_query_variants'],2)
        with self.assertRaises(ValueError):coverage.observe(plan|dict(split='val'))
        with self.assertRaises(ValueError):coverage.observe(plan|dict(query_id='v'))

    def test_crc_control_coverage_uses_only_its_configured_pool(self):
        corpus=SimpleNamespace(rows=[dict(tile_id='s',dataset='crc',split='train'),dict(tile_id='t',dataset='tiger',split='train')],eligible=[{0},{1}],index={'s':0,'t':1})
        coverage=Coverage(corpus,datasets=['crc'])
        self.assertEqual(coverage.summary()['query_pool_tiles'],1)
        self.assertEqual(set(coverage.summary()['by_dataset']),{'crc'})


if __name__=='__main__':unittest.main()
