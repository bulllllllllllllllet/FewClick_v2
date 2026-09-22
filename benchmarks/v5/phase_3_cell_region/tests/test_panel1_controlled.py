import unittest
import numpy as np
import torch
from benchmarks.v5.phase_3_cell_region.src.panel1_controlled import common_cells,spread,overlaps,interior_click,next_click,confusion_scores

class ControlledTest(unittest.TestCase):
    def test_common_contract_does_not_modify_source(self):
        x=torch.randn(5,74); old=x.clone(); y=common_cells(x)
        torch.testing.assert_close(x,old)
        self.assertEqual(int(torch.count_nonzero(y[:,2:10])),0)
        torch.testing.assert_close(y[:,:2],x[:,:2]); torch.testing.assert_close(y[:,10:],x[:,10:])
    def test_sampling_reaches_all_groups_without_replacement(self):
        rows=[dict(sample_id=f'{c}:{g}:{i}',case=c,group=g) for c in ('a','b','c') for g in range(2) for i in range(5)]
        s=spread(rows,12,('case','group'),17)
        self.assertEqual(len({r['sample_id'] for r in s}),12)
        self.assertEqual(len({(r['case'],r['group']) for r in s}),6)
        self.assertEqual(s,spread(list(reversed(rows)),12,('case','group'),17))
    def test_small_budget_balances_top_level_even_with_many_subgroups(self):
        rows=[dict(sample_id=f'{c}:{g}',case=c,group=g) for c in ('a','b','c') for g in range(30)]
        result=spread(rows,3,('case','group'),17)
        self.assertEqual({r['case'] for r in result},{'a','b','c'})
    def test_buffer_and_source_identity(self):
        a=dict(wsi_id='a',x_level0=0,y_level0=0,width_level0=100,height_level0=100)
        self.assertTrue(overlaps(a,dict(a,x_level0=110),20))
        self.assertFalse(overlaps(a,dict(a,x_level0=121),20))
        self.assertFalse(overlaps(a,dict(a,wsi_id='b'),20))
    def test_clicks_are_interior_valid_and_not_repeated(self):
        target=np.zeros((20,20),bool); target[3:15,4:17]=True
        y,x=interior_click(target); self.assertTrue(target[y,x])
        valid=np.ones_like(target); pred=np.zeros_like(target)
        nxt=next_click(pred,target,valid,[(y,x,1)])
        self.assertNotEqual(nxt[:2],(y,x)); self.assertTrue(target[nxt[:2]])
        self.assertIsNone(next_click(target,target,valid,[]))
    def test_confusion_metrics_penalize_foreground_miss(self):
        scores=confusion_scores(np.array([[90,0],[10,0]]))
        self.assertAlmostEqual(scores['pixel_accuracy'],.9)
        self.assertEqual(scores['by_class']['1']['dice'],0)
        self.assertLess(scores['macro_dice'],.5)

if __name__=='__main__': unittest.main()
