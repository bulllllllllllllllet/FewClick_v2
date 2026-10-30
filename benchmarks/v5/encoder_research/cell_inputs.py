"""Same retained nuclei/order and RGB information for continuous-cell branches."""
import hashlib
import json
import numpy as np
from PIL import Image
from scipy import ndimage


def load_continuous_cell_inputs(row):
    variant=json.loads(row['variants_json'])['original']
    assert hashlib.sha256(open(variant['cells'],'rb').read()).hexdigest()==variant['cells_sha256']
    assert hashlib.sha256(open(variant['image'],'rb').read()).hexdigest()==variant['image_sha256']
    with np.load(variant['cells'],allow_pickle=False) as z:
        inst=z['instances'].copy();meta=z['meta'].copy();reg=z['reg'].copy();features=z['features'].copy();total=int(z['total'])
    assert meta.shape==(len(reg),4) and reg.shape==(len(features),64) and features.shape==(len(meta),74)
    assert np.array_equal(features[:,10:],reg) and np.isfinite(features).all()
    ids=np.unique(inst);ids=ids[ids>0][:255]
    if len(ids):
        yx=np.asarray(ndimage.center_of_mass(np.ones_like(inst,np.uint8),inst,ids))
        order=np.lexsort((yx[:,1],yx[:,0]));ids=ids[order];yx=yx[order]
        assert len(ids)==len(meta)
        assert np.allclose(yx[:,::-1]/np.array(inst.shape[::-1]),meta[:,:2],atol=1e-6)
    else:assert len(meta)==0
    with Image.open(variant['image']) as im:
        rgb=np.asarray(im.convert('RGB').resize(inst.shape[::-1],Image.Resampling.BILINEAR)).copy()
    fine_mpp=np.asarray(json.loads(row['input_mpp_xy_json']),float)
    sampling_mpp=fine_mpp*512/np.asarray(inst.shape[::-1])
    w,h=np.rint(10/sampling_mpp).astype(int);crops=[];edge_padded=0
    for position in meta[:,:2]*np.asarray(inst.shape[::-1]):
        left,top=np.rint(position-np.array([w,h])/2).astype(int)
        right,bottom=left+w,top+h
        pad=((max(0,-top),max(0,bottom-rgb.shape[0])),(max(0,-left),max(0,right-rgb.shape[1])),(0,0))
        edge_padded+=int(any(v for pair in pad[:2] for v in pair))
        padded=np.pad(rgb,pad,mode='edge') if any(v for pair in pad[:2] for v in pair) else rgb
        x0,y0=left+pad[1][0],top+pad[0][0]
        crop=padded[y0:y0+h,x0:x0+w];assert crop.shape==(h,w,3)
        crops.append(Image.fromarray(crop).resize((40,40),Image.Resampling.BICUBIC))
    return dict(crops=crops,instance_ids=ids,positions=meta[:,:2],metadata10=features[:,:10],old_reg64=reg,
        total=total,edge_padded=edge_padded,actual_crop_um=[w*sampling_mpp[0],h*sampling_mpp[1]],
        native_information_mpp=fine_mpp.tolist(),sampling_mpp=sampling_mpp.tolist())
