import unittest
import pandas as pd
from benchmarks.v5.phase_3_cell_region.src.crc_region_supervision import STRUCTURE_CLASSES, aggregate_chunk, add_coverage, structure_class

class RegionSupervisionTest(unittest.TestCase):
 def test_unmapped_cells_remain_in_coverage_denominator(self):
  frame=pd.DataFrame({"x":[0,10,20],"y":[0,10,20],"class":["Epithelial","FAP_CAF","Undefined"]})
  table=add_coverage(aggregate_chunk(frame,structure_class,STRUCTURE_CLASSES,1.,100.),STRUCTURE_CLASSES,"panel1")
  self.assertEqual(int(table.panel1_observed_cells.iloc[0]),3); self.assertEqual(int(table.panel1_mapped_cells.iloc[0]),2)
  self.assertAlmostEqual(float(table.panel1_coverage.iloc[0]),2/3)
if __name__ == "__main__": unittest.main()
