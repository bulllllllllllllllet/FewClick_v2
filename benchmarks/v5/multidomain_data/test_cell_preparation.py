import unittest
import numpy as np

from .prepare_cells import filter_padding,balanced_selection


class CellPreparationTests(unittest.TestCase):
    def test_padding_filter_uses_image_extent_only(self):
        inst=np.zeros((512,512),np.int32);inst[10:20,10:20]=1;inst[200:210,200:210]=2
        metadata={'source_image_size':[256,256],'input_to_image_affine':[1,0,-128,0,1,-128]}
        result=filter_padding(inst,metadata)
        self.assertFalse((result==1).any());self.assertEqual(int((result==2).sum()),100)
        self.assertEqual(int((inst==1).sum()),100)

    def test_balancing_spans_quality_and_datasets(self):
        rows=[dict(dataset=d,organ='o',quality=q,selection_hash=str(i)) for d,q,n in [('a','original',10),('a','0.8',2),('b','original',2)] for i in range(n)]
        selected=balanced_selection(rows,3)
        self.assertEqual({(r['dataset'],r['quality']) for r in selected},{('a','original'),('a','0.8'),('b','original')})
        self.assertEqual(selected,balanced_selection(rows,3))


if __name__=='__main__':unittest.main()
