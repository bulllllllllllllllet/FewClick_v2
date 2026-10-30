"""Fixed common10um cell features; three encoders run in separate processes."""
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
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.time()
    manifest=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_cell_probe_manifest_1030_182424/cells.jsonl')
    rows=[json.loads(x) for x in manifest.read_text().splitlines()];assert len(rows)==570
    proof=json.loads((manifest.parent/'all570_decode_acceptance.json').read_text())
    expected={x['cell_id']:x['canonical_pixels_sha256'] for x in proof['cells']}
    status=dict(status='loading',arm=a.arm,worker_pid=os.getpid(),completed=0,total=len(rows))
    def save():
        status['elapsed_seconds']=time.time()-start
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();device='cuda:0'
    @lru_cache(maxsize=8)
    def image(path,identity):
        assert sha(path)==identity
        with Image.open(path) as im:return im.convert('RGB').copy()
    def crop(row):
        im=image(row['image'],row['image_sha256']);mx,my=row['mpp_xy'];x,y=row['xy']
        w,h=round(10/mx),round(10/my);left,top=round(x-w/2),round(y-h/2)
        patch=im.crop((left,top,left+w,top+h)).resize((40,40),Image.Resampling.BICUBIC)
        assert hashlib.sha256(np.asarray(patch).tobytes()).hexdigest()==expected[row['cell_id']]
        return patch
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
    identities={path:sha(path) for path in weight_paths}
    with torch.inference_mode():
        gate=[crop(x) for x in rows[:4]];first=encode(gate);repeat=encode(gate)
        assert first.shape==(4,dimension) and torch.isfinite(first).all() and torch.allclose(first,repeat,atol=1e-5,rtol=1e-5)
        (a.output/'input_gate.json').write_text(json.dumps(dict(status='passed_real_common_crop_gate',shape=list(first.shape),repeat_max_abs_error=float((first-repeat).abs().max()),pixel_hashes_checked=True))+'\n')
        results=[]
        for offset in range(0,len(rows),16):
            values=encode([crop(r) for r in rows[offset:offset+16]]).float().cpu().numpy()
            assert values.shape==(len(rows[offset:offset+16]),dimension) and np.isfinite(values).all()
            results.append(values);status.update(status='running',completed=offset+len(values));save()
    values=np.concatenate(results)
    np.savez_compressed(a.output/'features.npz',features=values,cell_ids=np.array([r['cell_id'] for r in rows]),
        splits=np.array([r['split'] for r in rows]),labels=np.array([r['label'] for r in rows]),organs=np.array([r['organ'] for r in rows]))
    report=dict(status='complete_features_unaccepted',arm=a.arm,worker_pid=os.getpid(),shape=list(values.shape),
        manifest_sha256=sha(manifest),source_sha256=sha(__file__),weights=identities,loading=loading,
        feature_sha256=sha(a.output/'features.npz'),elapsed_seconds=time.time()-start,
        physical_view_um=10,point_location_input='same expert center for all arms; no class labels enter encoder',
        qualification='CTransPath center-crop control is not original mask/context XCell; classification and downstream outcomes not yet evaluated')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete',completed=len(rows));save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
