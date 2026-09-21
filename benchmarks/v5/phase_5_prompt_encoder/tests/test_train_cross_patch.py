import unittest
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts
class CrossTrainTests(unittest.TestCase):
 def test_gap_is_required(self): self.assertTrue(separated_contexts((0,0,10,10),(522,0,532,10)))
