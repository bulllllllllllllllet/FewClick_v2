import unittest
import numpy as np
from .prepare import normalize_source_labels

from .prepare_cells import filter_padding,balanced_selection,shard_selection


class CellPreparationTests(unittest.TestCase):
    def test_source_void_255_is_preserved_without_accepting_unknown_classes(self):
        labels=np.array([[0,1,16,255]],np.uint8)
        np.testing.assert_array_equal(normalize_source_labels(labels,'ignite'),[[255,1,16,255]])
        self.assertEqual(int(labels[0,0]),0)
        with self.assertRaises(ValueError):normalize_source_labels(np.array([[254]],np.uint8),'ignite')

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

    def test_shards_cover_inputs_once_independent_of_input_order(self):
        rows=[dict(tile_id=str(i),quality=q) for i in range(17) for q in ['original','0.8','1.0']]
        groups=[shard_selection(rows,i,6) for i in range(6)]
        keys=[(r['tile_id'],r['quality']) for group in groups for r in group]
        self.assertEqual(len(keys),len(set(keys)));self.assertEqual(len(keys),len(rows))
        self.assertEqual(groups[2],shard_selection(rows[::-1],2,6))
        with self.assertRaises(ValueError):shard_selection(rows+rows[:1],0,6)
        with self.assertRaises(ValueError):shard_selection(rows,6,6)


if __name__=='__main__':unittest.main()
