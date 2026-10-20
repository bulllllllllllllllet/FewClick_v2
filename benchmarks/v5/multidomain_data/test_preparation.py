"""Geometry/leakage behavior, independent of model outputs."""
import json
import unittest

import numpy as np

from .geometry import rasterize_tissue, ring_mask, spatial_split, standardize, quality_version
from .prepare import tile_catalog


def feature(label, points, holes=()):
    return {'properties':{'classification':{'name':label}},
            'geometry':{'type':'Polygon','coordinates':[points,*holes]}}


class PreparationTests(unittest.TestCase):
    def test_polygon_centers_holes_and_clipping(self):
        exterior=[[0,0],[4,0],[4,4],[0,4],[0,0]]
        hole=[[1,1],[3,1],[3,3],[1,3],[1,1]]
        doc={'features':[feature('tumor',exterior,[hole])]}
        labels,_=rasterize_tissue(doc,(4,4),{'tumor':1,'tissue_white_background':6})
        self.assertEqual(int((labels==1).sum()),12)
        self.assertTrue((labels[1:3,1:3]==255).all())
        self.assertFalse(ring_mask([[-8,0],[-4,0],[-4,4],[-8,4],[-8,0]],(4,4)).any())

    def test_overlap_is_ignore_and_background_is_not_negative_fill(self):
        a=[[0,0],[3,0],[3,3],[0,3],[0,0]]
        b=[[2,2],[4,2],[4,4],[2,4],[2,2]]
        background=[[0,0],[4,0],[4,4],[0,4],[0,0]]
        f=[feature('tumor',a),feature('stroma',b)]
        names={'tumor':1,'stroma':2,'tissue_white_background':6}
        labels,audit=rasterize_tissue({'features':f},(5,5),names)
        self.assertEqual(labels[2,2],255);self.assertEqual(audit['conflict_pixels'],1)
        self.assertEqual(labels[0,4],255)
        labels,_=rasterize_tissue({'features':[feature('tissue_white_background',background),*f]},(5,5),names)
        self.assertEqual(labels[0,0],1);self.assertEqual(labels[3,0],6);self.assertEqual(labels[2,2],255)

    def test_label_grid_and_padding_preserve_physical_coordinates(self):
        rgb=np.arange(48,dtype=np.uint8).reshape(4,4,3)
        gt=np.array([[1,2],[3,4]],np.uint8)
        image,labels,meta=standardize(rgb,gt,[0,0,4,4],[.25,.25],size=4)
        np.testing.assert_array_equal(labels,np.repeat(np.repeat(gt,2,0),2,1))
        np.testing.assert_array_equal(image,rgb)
        self.assertEqual(meta['input_mpp_xy'],[.25,.25])
        image,labels,_=standardize(rgb,gt,[-1,-1,6,6],[.25,.25],size=6)
        self.assertTrue((labels[0]==255).all());self.assertTrue((image[0]==255).all())
        self.assertEqual(int((labels!=255).sum()),16)

    def test_quality_preserves_shape_and_does_not_invent_sharper_source(self):
        yy,xx=np.indices((512,512));v=((xx+yy)%2*255).astype(np.uint8)
        rgb=np.repeat(v[...,None],3,2)
        degraded,meta=quality_version(rgb,[.5,.5],[.25,.25],'0.8')
        self.assertEqual(degraded.shape,rgb.shape);self.assertEqual(meta['intermediate_size'],[320,320])
        self.assertLess(float(degraded.std()),float(rgb.std())/4)
        _,meta=quality_version(rgb,[.5,.5],[1.0,1.0],'0.8')
        self.assertEqual(meta['intermediate_size'],[256,256])

    def test_degraded_padding_stays_white(self):
        rgb=np.zeros((400,400,3),np.uint8);gt=np.ones((400,400),np.uint8)
        image,_,meta=standardize(rgb,gt,[-56,-56,512,512],[.5,.5])
        result,_=quality_version(image,[.5,.5],[.5,.5],'0.8',meta)
        self.assertTrue((result[:56]==255).all());self.assertTrue((result[:,:56]==255).all())

    def test_padding_does_not_create_dark_rim_in_valid_white_image(self):
        rgb=np.full((100,100,3),255,np.uint8);gt=np.ones((100,100),np.uint8)
        image,_,_=standardize(rgb,gt,[-10,-10,120,120],[.5,.5],size=512)
        self.assertTrue((image==255).all())

    def test_tiger_duplicate_source_footprints_share_split(self):
        def source(identity,group):
            return dict(source_id=identity,dataset='tiger',organ='breast',split='spatial_pending',slide_id='slide',slide_id_verified=True,
                width=1024,height=1024,available_mpp_x=.5,available_mpp_y=.5,
                extra_json=json.dumps({'slide_origin_xy':[128,128],'source_group':group}))
        rows,_,duplicates=tile_catalog([source('big','tissue-bcss'),source('small','tissue-cells')])
        self.assertEqual(len(rows),1);self.assertEqual(len(duplicates),1)
        self.assertEqual(rows[0]['source_id'],'small')
        self.assertIsNone(spatial_split('slide',[0,0,512,512]))
        self.assertIsNotNone(spatial_split('slide',[512,512,512,512]))

    def test_puma_holdout_and_crc_existing_group_are_preserved(self):
        common=dict(organ='o',slide_id=None,slide_id_verified=False,width=1024,height=1024,available_mpp_x=.25,available_mpp_y=.25,split='train')
        p=dict(common,source_id='p',dataset='puma',extra_json='{}')
        c=dict(common,source_id='c',dataset='crc',extra_json=json.dumps({'spatial_group_id':'original-block'}))
        rows,_,_=tile_catalog([p,c]);by={r['dataset']:r for r in rows}
        self.assertFalse(by['puma']['training_allowed']);self.assertEqual(by['crc']['spatial_group_id'],'original-block')
        self.assertEqual(json.loads(by['puma']['image_box_json']),[0,0,1024,1024])


if __name__=='__main__':unittest.main()
