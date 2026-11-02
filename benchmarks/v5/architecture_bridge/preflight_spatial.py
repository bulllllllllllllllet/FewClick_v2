"""Fail-closed P3 environment and receipt audit; never launches training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import pyvips
import torch
import transformers
from transformers import AutoConfig


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--learning',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--smoke-root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);checks={};bindings={}
    def receipt(label,path,status):
        if not path.exists():checks[label]='pending';return None
        obj=json.loads(path.read_text());assert obj['status']==status,(label,obj['status'])
        bindings[str(path)]=sha(path);checks[label]='accepted';return obj
    environment=dict(python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,transformers_source=transformers.__file__,
                     pyvips=pyvips.__version__,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    allowed_visible=environment['CUDA_VISIBLE_DEVICES'] in ['1','3','4','5']
    env_ok=transformers.__version__=='4.49.0' and environment['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and allowed_visible
    checks['environment']='accepted' if env_ok else 'invalid'
    if transformers.__version__=='4.49.0':
        config=AutoConfig.from_pretrained('/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2',local_files_only=True)
        assert config.model_type=='dinov2';checks['local_phikon_config']='accepted'
    else:checks['local_phikon_config']='unverified_in_wrong_environment'
    cache=receipt('normalized_inputs',a.cache/'acceptance.json','accepted_repaired_normalized_cells')
    assert cache and cache['validation_fitting_cells']==0
    for file,key in [('index.jsonl','index_sha256'),('transforms.npz','transforms_sha256')]:assert sha(a.cache/file)==cache[key]
    protocol=receipt('task_protocol',a.protocol/'acceptance.json','accepted_spatial_task_protocol_only');assert protocol
    assert protocol['input_identity']==dict(repaired_index_sha256=cache['index_sha256'],transforms_sha256=cache['transforms_sha256'])
    assert sha(a.protocol/'training_consumption.jsonl')==protocol['consumption_sha256'] and sha(a.protocol/'evaluation.jsonl')==protocol['evaluation_sha256']
    for arm in ['S11','I','S00','H11','S10','S01']:
        proof=receipt('learning_'+arm,a.learning/arm/'independent_acceptance.json','accepted_P3_spatial_train_learning_gate')
        if proof:
            assert proof['arm']==arm and proof['seed']==20261103 and proof['masks_replayed']==32
            assert proof['report_sha256']==sha(a.learning/arm/'report.json')
            assert proof['masks_sha256']==sha(a.learning/arm/'masks.npz')
            assert proof['checkpoint_sha256']==sha(a.learning/arm/'checkpoint.pt')
        proof=receipt('smoke_'+arm,a.smoke_root/(arm+'_smoke')/'independent_acceptance.json','accepted_P3_spatial_evaluation')
        if proof:
            assert proof['arm']==arm and proof['smoke'] and proof['rows']==12
            assert proof['report_sha256']==sha(a.smoke_root/(arm+'_smoke')/'report.json')
    inventory=receipt('inventory',a.inventory,'accepted_full_spatial_inventory')
    if inventory:
        assert inventory['fields']==1737 and inventory['views']==5211 and inventory['all_image_views_pixel_exact']
        assert inventory['identity']['index_sha256']==cache['index_sha256']
        assert inventory['frames_sha256']==sha(a.inventory.parent/'frames.jsonl')
    # These requirements deliberately stay explicit until their implementations have real evidence.
    missing_integration=['formal_train_monitor_fixed_nodes','GPU20update100mask_benchmark','frozen_execution_contract_and_source_binding']
    result=dict(status='P3_preflight_not_ready',environment=environment,checks=checks,evidence=bindings,
                sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
                remaining_integration=missing_integration,training_allowed=False,
                qualification='environment/config and receipts only; real forward/backward and optimizer acceptance recorded separately')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
