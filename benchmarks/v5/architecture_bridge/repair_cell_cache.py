"""Rebuild selected native XCell sets; persist full masks for future audits."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import pyvips
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from .repaired_cell_geometry import cell_geometry, POLICY
from .deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS, sha
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.multidomain_data.fast_nuclei import CachedCenterCropper, InstanSegDetector, DEFAULT_INSTANSEG_CHECKPOINT
from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models, WEIGHTS, attended_crop
from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify_geometry(inst, meta, ids, key, mpp):
    all_ids = np.unique(inst); all_ids = all_ids[all_ids > 0]
    expected = sorted(all_ids.tolist(), key=lambda n: hashlib.sha256(f'{POLICY}|{key}|{n}'.encode()).hexdigest())[:255]
    assert set(ids.tolist()) == set(expected) and len(ids) == min(255, len(all_ids))
    boxes = ndimage.find_objects(inst)
    for label, row in zip(ids, meta):
        sy, sx = boxes[int(label)-1]
        y, x = np.nonzero(inst[sy, sx] == label)
        xy = np.array([(x.mean()+sx.start+.5)/inst.shape[1], (y.mean()+sy.start+.5)/inst.shape[0]], np.float32)
        assert np.array_equal(xy, row[:2])
        assert np.isclose(row[2], np.log1p(len(x)*np.prod(mpp)/.25**2), atol=1e-6)
    # A linear pixel-center coordinate field must sample the requested point.
    if len(ids):
        h, w = 64, 64
        yy, xx = torch.meshgrid((torch.arange(h)+.5)/h, (torch.arange(w)+.5)/w, indexing='ij')
        field = torch.stack((xx, yy))[None]
        got = sample_assignment_at_cells(field, torch.from_numpy(meta)[None])[0].T.numpy()
        expected_xy = np.clip(meta[:, :2], [.5/w, .5/h], [1-.5/w, 1-.5/h])
        assert np.allclose(got, expected_xy, atol=2e-7)


@torch.inference_mode()
def encode(rgb, inst, ids, ctp, xcell):
    if not len(ids): return np.empty((0, 64), np.float32)
    cropper = CachedCenterCropper(); tensors = []
    for j, label in enumerate(ids):
        crop = cropper(rgb, inst, int(label))
        if j in {0, len(ids)//2, len(ids)-1}:
            assert np.array_equal(crop, attended_crop(rgb, inst, int(label)))
        value = torch.from_numpy(crop.copy()).permute(2,0,1).float()[None]/255
        value = F.interpolate(value, size=(224,224), mode='bilinear', align_corners=False, antialias=True)[0]
        tensors.append((value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
    raw = torch.cat([ctp(torch.stack(tensors[i:i+32]).cuda()).float() for i in range(0,len(ids),32)])
    padded = raw.new_zeros((1,255,768)); valid = raw.new_zeros((1,255))
    padded[0,:len(ids)] = raw; valid[0,:len(ids)] = 1
    return xcell(None,padded,valid)[1][0,:len(ids)].float().cpu().numpy()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true'); p.add_argument('--gate-receipt',type=Path)
    p.add_argument('--accept',action='store_true'); a=p.parse_args()
    torch.set_num_threads(2)
    catalog, inventory=load_cell_catalog(TASKS)
    selected=[min((r for r in inventory.values() if r['dataset']==d and r['split']==s),key=lambda r:r['tile_id']) for d in ['crc','tiger'] for s in ['train','val']] if a.gate else [inventory[k] for k in sorted(inventory)]
    identity=dict(policy=POLICY,catalog_sha256=sha(catalog/'tiles.parquet'),
                  sources={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('repaired_cell_geometry.py')):sha(Path(__file__).with_name('repaired_cell_geometry.py'))},
                  weights={k:sha(v) for k,v in WEIGHTS.items() if k!='cpsam'}|{'instanseg':sha(DEFAULT_INSTANSEG_CHECKPOINT)})
    if a.accept:
        report=json.loads((a.output/'report.json').read_text()); assert report['identity']==identity
        assert report['gate']==a.gate and len(report['frames'])==len(selected)
        for frame,row in zip(report['frames'],selected):
            assert frame['tile_id']==row['tile_id'] and sha(frame['file'])==frame['sha256']
            with np.load(frame['file'],allow_pickle=False) as f:
                for scale in range(3):
                    inst,meta,ids,mpp=[f[f'{k}_{scale}'] for k in ['instances','meta','ids','mpp']]
                    verify_geometry(inst,meta,ids,row['tile_id']+'|'+str(scale),mpp)
                    assert np.array_equal(encode_xcell_features(meta,f[f'reg_{scale}']),f[f'cells_{scale}'])
                    assert np.isfinite(f[f'cells_{scale}']).all()
        if a.gate: assert report['fresh_reload_max_abs'] <= 1e-5
        dump(a.output/'acceptance.json',dict(status='accepted_repaired_cell_gate' if a.gate else 'accepted_repaired_raw_cells',
             report_sha256=sha(a.output/'report.json'),identity=identity,frames=len(selected),normalization_accepted=False,training_allowed=False))
        return
    if not a.gate:
        assert a.gate_receipt
        gate=json.loads(a.gate_receipt.read_text()); assert gate['status']=='accepted_repaired_cell_gate' and gate['identity']==identity
    a.output.mkdir(parents=True,exist_ok=False)
    dump(a.output/'contract.json',dict(identity=identity,gate=a.gate,context_masks='explicitly redetected same detector; old masks unavailable',GT_used=False))
    ctp,xcell,_=load_feature_models(torch.device('cuda:0'))
    detector=InstanSegDetector(DEFAULT_INSTANSEG_CHECKPOINT,torch.device('cuda:0'))
    records=[]; first=None; started=time.time()
    try:
        for row in selected:
            arrays={}; variant=json.loads(row['variants_json'])['original']
            assert sha(variant['image'])==variant['image_sha256'] and sha(variant['cells'])==variant['cells_sha256']
            assert sha(row['multiscale_cache'])==row['multiscale_sha256']
            for scale in range(3):
                if scale==0:
                    with np.load(variant['cells'],allow_pickle=False) as f: inst=f['instances'].copy()
                    with Image.open(variant['image']) as im: rgb=np.asarray(im.convert('RGB')).copy()
                    mpp=np.asarray(json.loads(row['input_mpp_xy_json']))*512/np.array(inst.shape[::-1])
                    high=np.asarray(Image.fromarray(rgb).resize(inst.shape[::-1],Image.Resampling.BILINEAR)).copy()
                else:
                    rgb=native_context(row,scale); native_mpp=np.asarray(json.loads(row['multiscale_json'])['native_mpp'])
                    size=tuple(round(rgb.shape[1-i]*native_mpp[i]/.25) for i in range(2))
                    high=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)).copy()
                    mpp=np.array([rgb.shape[1-i]*native_mpp[i]/size[i] for i in range(2)])
                    inst=np.asarray(detector.eval(high,input_mpp=mpp)[0],np.int32)
                meta,ids,total=cell_geometry(inst,row['tile_id']+'|'+str(scale),mpp)
                verify_geometry(inst,meta,ids,row['tile_id']+'|'+str(scale),mpp)
                reg=encode(high,inst,ids,ctp,xcell)
                assert reg.shape==(len(ids),64) and np.isfinite(reg).all()
                if first is None and a.gate and len(ids):first=(high.copy(),inst.copy(),ids.copy(),reg.copy())
                for k,v in dict(instances=inst,meta=meta,ids=ids,reg=reg,cells=encode_xcell_features(meta,reg),total=np.array(total),mpp=mpp,
                                image=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR))).items():arrays[f'{k}_{scale}']=v
            path=a.output/(row['tile_id']+'.npz'); np.savez_compressed(path,**arrays)
            records.append(dict(tile_id=row['tile_id'],split=row['split'],dataset=row['dataset'],file=str(path),sha256=sha(path)))
            dump(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),completed=len(records),total=len(selected),seconds=time.time()-started))
            print(json.dumps(dict(completed=len(records),total=len(selected),seconds=time.time()-started)),flush=True)
        error=None
        if a.gate:
            del ctp,xcell; gc.collect();torch.cuda.empty_cache();ctp,xcell,_=load_feature_models(torch.device('cuda:0'))
            high,inst,ids,expected=first; got=encode(high,inst,ids,ctp,xcell)
            error=float(np.max(np.abs(got-expected))); assert np.allclose(got,expected,atol=1e-5,rtol=1e-5)
        dump(a.output/'report.json',dict(identity=identity,frames=records,gate=a.gate,fresh_reload_max_abs=error,GT_used=False,worker_pid=os.getpid()))
        dump(a.output/'status.json',dict(status='complete_unaccepted',worker_pid=os.getpid(),completed=len(records)))
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),completed=len(records),error=repr(exc)));raise


if __name__=='__main__':main()
