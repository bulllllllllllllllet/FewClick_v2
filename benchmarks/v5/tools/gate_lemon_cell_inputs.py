"""Pinned official LEMON builder, strict weights and real input/reload gate."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.time()
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/encoder_inventory_1030_182424/LEMON')
    expected='946199ff11d3a358bbea531aced36ff02b49648f8b7fd752b1b6d8c772a84d26'
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==expected
    receipt=json.loads(Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/lemon_weights_1031_004350/data/status.json').read_text())
    assert receipt['status']=='complete_download_identity_only'
    spec=importlib.util.spec_from_file_location('pinned_lemon_official',source/'model.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    stats=json.loads((source/'mean_std.json').read_text())
    transform=Compose([ToTensor(),Resize((40,40)),Normalize(stats['mean'],stats['std'])])
    contract=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_cell_probe_manifest_1030_182424/contract.json')
    selected=json.loads(contract.read_text())['gates'][:4]
    batch=torch.stack([transform(Image.open(x['crop']).convert('RGB')) for x in selected]).to('cuda:0')
    outputs=[]
    for repeat in range(2):
        checkpoint=torch.load('/nfs-medical3/zyh/fewclick_v2/cvpr_research/lemon_weights_1031_004350/data/lemon_vits8.pth.tar',map_location='cpu',weights_only=True)
        model=official.build_lemon_moco_backbone('vits8')
        model.load_state_dict(official.extract_backbone_state_dict(checkpoint),strict=True)
        del checkpoint
        model=model.to('cuda:0').eval().requires_grad_(False)
        with torch.inference_mode():
            tokens=model.forward_features(batch);features=model(batch)
            assert tuple(tokens.shape)==(4,26,384)
            assert tuple(features.shape)==(4,384) and torch.isfinite(features).all()
            outputs.append(features.cpu().numpy())
        del model,tokens,features;gc.collect();torch.cuda.empty_cache()
    error=float(np.max(np.abs(outputs[0]-outputs[1])));assert np.allclose(*outputs,rtol=1e-5,atol=1e-5)
    np.savez_compressed(a.output/'features.npz',features=outputs[0],cell_ids=np.array([x['cell_id'] for x in selected]))
    report=dict(status='complete_LEMON_real_input_strict_load_reload_gate',worker_pid=os.getpid(),
        official_model_code_sha256=expected,weights_sha256=receipt['sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),selected=selected,
        input_shape=[4,3,40,40],token_shape=[4,26,384],features_shape=[4,384],
        dtype='float32',physical_view='approximately10um at0.25MPP canonical crop',
        reload_max_abs_error=error,elapsed_seconds=time.time()-start,training_or_quality_result=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
