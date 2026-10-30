"""Verify real frozen Phikon cell input, spatial tokens and fresh model reload."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import transformers
from benchmarks.v5.phase_3_cell_region.src.phikon_v2 import PhikonV2Encoder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);start=time.time()
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_cell_probe_manifest_1030_182424/contract.json')
    contract=json.loads(source.read_text());selected=contract['gates'][:4]
    images=[Image.open(x['crop']).convert('RGB') for x in selected]
    weights=Path('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2')
    outputs=[]
    for repeat in range(2):
        model=PhikonV2Encoder(weights,device=a.device);model.freeze_backbone()
        batch=model.preprocess(images,a.device)
        assert tuple(batch.shape)==(4,3,224,224)
        with torch.inference_mode():
            tokens=model.backbone(pixel_values=batch).last_hidden_state
            assert tuple(tokens.shape)==(4,197,1024) and torch.isfinite(tokens).all()
            ids=[1+y*14+x for y in (6,7) for x in (6,7)]
            features=torch.cat([tokens[:,0],tokens[:,ids].mean(1)],dim=1).cpu().numpy()
        outputs.append(features)
        del model,batch,tokens
        gc.collect()
        if str(a.device).startswith('cuda'):torch.cuda.empty_cache()
    error=float(np.max(np.abs(outputs[0]-outputs[1])));assert np.allclose(*outputs,rtol=1e-5,atol=1e-5)
    np.savez_compressed(a.output/'features.npz',features=outputs[0],cell_ids=np.array([x['cell_id'] for x in selected]))
    report=dict(status='complete_frozen_input_spatial_token_reload_gate',cells=4,device=a.device,
        transformer_version=transformers.__version__,input_shape=[4,3,224,224],token_shape=[4,197,1024],
        features_shape=[4,2048],feature_definition='concatenated CLS1024 and center4 spatial-token mean1024, diagnostic only',
        reload_max_abs_error=error,elapsed_seconds=time.time()-start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        crop_contract_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        physical_view='approximately10um canonical40px resized by official224px preprocessor; interpolated detail disclosed',
        training_or_quality_result=False,selected=selected)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
