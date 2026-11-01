"""Fixed synthetic color features; identity gate before every stress condition."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose,ToTensor,Resize,Normalize


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['ctranspath','lemon','phikon'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);device='cuda:0'
    root=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_color_inputs_1101_122151/data')
    proof=json.loads((root/'acceptance.json').read_text())
    assert proof['status']=='accepted_186cell_930color_variant_pixels'
    assert sha(root/'variants.jsonl')==proof['variants_sha256']
    rows=[json.loads(s) for s in (root/'variants.jsonl').read_text().splitlines()]
    conditions=['identity','OD_RG_plus','OD_RG_minus','OD_light','OD_dark']
    rows=[r for c in conditions for r in rows if r['condition']==c];assert len(rows)==930
    def crop(row):
        assert sha(row['crop'])==row['file_sha256']
        with Image.open(row['crop']) as im: image=im.convert('RGB').copy()
        assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest()==row['pixel_sha256']
        return image
    if a.arm=='ctranspath':
        from benchmarks.v5.phase_6_mask_decoder.external_cell_inputs import load_feature_models
        model,unused,loading=load_feature_models(torch.device(device));del unused
        dimension=768
        def encode(images):
            x=torch.stack([ToTensor()(im) for im in images]).to(device)
            x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            return model(x)
        weight_paths=['/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth']
    elif a.arm=='lemon':
        source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/encoder_inventory_1030_182424/LEMON')
        assert sha(source/'model.py')=='946199ff11d3a358bbea531aced36ff02b49648f8b7fd752b1b6d8c772a84d26'
        spec=importlib.util.spec_from_file_location('pinned_lemon',source/'model.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        path='/nfs-medical3/zyh/fewclick_v2/cvpr_research/lemon_weights_1031_004350/data/lemon_vits8.pth.tar'
        state=torch.load(path,map_location='cpu',weights_only=True);model=official.build_lemon_moco_backbone('vits8')
        model.load_state_dict(official.extract_backbone_state_dict(state),strict=True);del state
        model=model.to(device).eval().requires_grad_(False);stats=json.loads((source/'mean_std.json').read_text())
        transform=Compose([ToTensor(),Resize((40,40)),Normalize(stats['mean'],stats['std'])]);dimension=384;loading={'strict':True}
        def encode(images):return model(torch.stack([transform(im) for im in images]).to(device))
        weight_paths=[path]
    else:
        from benchmarks.v5.phase_3_cell_region.src.phikon_v2 import PhikonV2Encoder
        model=PhikonV2Encoder('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2',device=device);model.freeze_backbone()
        dimension=1024;loading={'feature':'mean center4 spatial tokens, not CLS'}
        def encode(images):
            tokens=model.backbone(pixel_values=model.preprocess(images,device)).last_hidden_state
            assert tuple(tokens.shape[1:])==(197,1024)
            return tokens[:,[1+y*14+x for y in (6,7) for x in (6,7)]].mean(1)
        weight_paths=['/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2/model.safetensors']

    weight_identities={p:sha(p) for p in weight_paths}
    old_root=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_features_1031_012400')/a.arm
    prior_report=json.loads((old_root/'report.json').read_text())
    assert weight_identities==prior_report['weights']
    old=np.load(old_root/'features.npz',allow_pickle=False)
    lookup={str(identity):i for i,identity in enumerate(old['cell_ids'])}
    features=[];timing={}
    with torch.inference_mode():
        for condition in conditions:
            subset=[r for r in rows if r['condition']==condition];assert len(subset)==186
            warm=encode([crop(subset[0])]);assert torch.isfinite(warm).all()
            torch.cuda.synchronize();start=time.perf_counter();values=[]
            for offset in range(0,len(subset),16):
                value=encode([crop(r) for r in subset[offset:offset+16]]).float().cpu().numpy()
                assert value.shape==(len(subset[offset:offset+16]),dimension) and np.isfinite(value).all()
                values.append(value)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            values=np.concatenate(values);timing[condition]=dict(seconds=elapsed,cells_per_second=186/elapsed,scope='warm encoder plus PNG SHA/decode/preprocessing/device transfer; excludes loading and segmentation')
            if condition=='identity':
                expected=old['features'][[lookup[r['cell_id']] for r in subset]]
                error=float(np.max(np.abs(values-expected)))
                assert np.allclose(values,expected,rtol=1e-5,atol=1e-5),error
                (a.output/'identity_gate.json').write_text(json.dumps(dict(status='accepted_identity_features',max_abs_error=error,cells=186))+'\n')
            features.append(values)
            print(json.dumps(dict(condition=condition,cells=186,seconds=elapsed)),flush=True)
    values=np.concatenate(features)
    np.savez_compressed(a.output/'features.npz',features=values,cell_ids=np.array([r['cell_id'] for r in rows]),conditions=np.array([r['condition'] for r in rows]))
    report=dict(status='complete_color_features_unaccepted',worker_pid=os.getpid(),arm=a.arm,shape=list(values.shape),
                source_sha256=sha(__file__),variants_sha256=proof['variants_sha256'],weights=weight_identities,
                prior_feature_sha256=sha(old_root/'features.npz'),feature_sha256=sha(a.output/'features.npz'),timing=timing,
                qualification='synthetic RGB color sensitivity; CTransPath point control is not original XCell; no probe quality evaluated yet')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
