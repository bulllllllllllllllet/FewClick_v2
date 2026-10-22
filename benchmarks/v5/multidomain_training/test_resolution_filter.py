"""Training exclusions must not alter held-fixed validation protocols."""
import json
import unittest

import pyvips
import numpy as np

from .episodes import EpisodeCorpus
from .resolution_filter import resolution_decision


class ResolutionPolicyTests(unittest.TestCase):
    def evidence(self, native=.5, quality=.5):
        return dict(available_mpp_xy_json=json.dumps([native, native]),
                    input_mpp_xy_json='[0.5,0.5]',
                    preparation_json=json.dumps({'effective_quality_mpp_xy': [quality, quality]}))

    def test_interpolation_cannot_hide_coarse_native_detail(self):
        self.assertFalse(resolution_decision(self.evidence(.778), .6)[0])
        self.assertFalse(resolution_decision(self.evidence(.25, 1.), .6)[0])
        self.assertTrue(resolution_decision(self.evidence(.25), .6)[0])
        for value in (float('nan'), 0., -1.):
            with self.assertRaises(ValueError):
                resolution_decision(self.evidence(value), .6)

    def corpus(self):
        corpus = EpisodeCorpus.__new__(EpisodeCorpus)
        corpus.seed = 123; corpus.prompt_policy = 'representative'
        corpus.rows = [dict(tile_id='a', organ='colon', input_mpp_xy_json='[0.5,0.5]', slide_verified=True)]
        corpus.index = {'a': 0}
        pool = {'local': {'colon': {1: [0]}}}
        corpus.pools = {split: {d: pool for d in ('crc', 'ocelot')} for split in ('train', 'val')}
        labels = np.zeros((64,64), np.uint8); labels[:,:32] = 1
        corpus.labels = lambda i: labels
        return corpus

    def test_excluded_domain_is_rejected_and_quality_cannot_reenter(self):
        corpus = self.corpus()
        corpus.training_policy = dict(datasets=['crc'], probabilities=[1.], qualities=['original'])
        for n in range(12):
            plan = corpus.plan(n, kind='local', quality_aug=True)
            self.assertEqual(plan['dataset'], 'crc')
            self.assertEqual((plan['support_quality'], plan['query_quality']), ('original','original'))
        with self.assertRaises(ValueError):
            corpus.plan(1, dataset='ocelot', kind='local')

    def test_policy_preserves_validation_task_coordinates_and_qualities(self):
        corpus = self.corpus()
        before = [corpus.plan(n, split='val', dataset='ocelot', kind='local') for n in range(12)]
        corpus.training_policy = dict(datasets=['crc'], probabilities=[1.], qualities=['original'])
        after = [corpus.plan(n, split='val', dataset='ocelot', kind='local') for n in range(12)]
        self.assertEqual(before, after)
        self.assertTrue(any(p['support_quality']!='original' for p in after))

    def test_materialize_rejects_forced_degraded_training_before_io(self):
        corpus = self.corpus()
        corpus.training_policy = dict(datasets=['crc'], probabilities=[1.], qualities=['original'])
        plan = corpus.plan(1, dataset='crc', kind='local')
        for key in ('support_quality', 'query_quality'):
            with self.assertRaisesRegex(ValueError, 'excluded low-resolution'):
                corpus.materialize(plan | {key: '0.8'}, 'cpu')


if __name__ == '__main__':
    unittest.main()
