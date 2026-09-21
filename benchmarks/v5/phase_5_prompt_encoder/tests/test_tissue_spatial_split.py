import unittest
from benchmarks.v5.phase_5_prompt_encoder.src.tissue_spatial_split import assign_context


def check_boundary_and_coarse_context():
    assert assign_context('a', [(300, 300, 512, 512)])
    assert assign_context('a', [(300, 300, 512, 512), (-100, -100, 4096, 4096)]) is None
    assert assign_context('a', [(16000, 300, 512, 512)]) is None


def check_same_block_group_and_determinism():
    a = assign_context('a', [(1000, 1000, 512, 512)])
    b = assign_context('a', [(3000, 3000, 512, 512)])
    assert (a['split'], a['spatial_group_id']) == (b['split'], b['spatial_group_id'])
    assert a == assign_context('a', [(1000, 1000, 512, 512)])


class SpatialSplitTests(unittest.TestCase):
    def test_boundary(self):
        check_boundary_and_coarse_context()

    def test_determinism(self):
        check_same_block_group_and_determinism()

    def test_bad_geometry(self):
        with self.assertRaises(ValueError):
            assign_context('a', [(0, 0, float('nan'), 10)])
        with self.assertRaises(ValueError):
            assign_context('a', [(0, 0, 10, 10)], block_size=1, margin=1)
