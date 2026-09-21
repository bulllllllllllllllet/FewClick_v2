import unittest
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts, transfer_support


class CrossPatchTests(unittest.TestCase):
    def test_full_context_gap(self):
        self.assertTrue(separated_contexts((0,0,4096,4096), (4608,0,8704,4096)))
        self.assertFalse(separated_contexts((0,0,4096,4096), (4500,0,8596,4096)))
        self.assertFalse(separated_contexts((0,0,4096,4096), (2000,2000,6096,6096)))

    def test_transfer_does_not_copy_labels_or_query_geometry(self):
        query = dict(binary_target='query_gt', region_xy='query_frame', fine_tokens='query_tokens')
        support = {f'{s}_{k}':f'{s}_{k}' for s in ('positive','negative') for k in ('tokens','xy','mask')}
        support.update(prompt_size_id=1, binary_target='support_gt', region_xy='support_frame')
        out = transfer_support(query, support)
        self.assertEqual(out['binary_target'], 'query_gt')
        self.assertEqual(out['region_xy'], 'query_frame')
        self.assertEqual(out['fine_tokens'], 'query_tokens')
        self.assertEqual(out['positive_xy'], 'positive_xy')
