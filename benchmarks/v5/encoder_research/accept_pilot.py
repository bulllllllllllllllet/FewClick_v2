"""Recount every saved prediction against immutable GT and frozen task identity."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from .shared_prompt import SharedPromptDecoder
from .image_features import CHANNELS
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.transfer_comparison.data import TransferData


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--remote-selection',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.pilot/'report.json').read_text());rows=json.loads((a.pilot/'rows.json').read_text())
    assert report['training_updates']==1000 and report['training_tasks']==4000 and report['exact_final_reload']
    training=Path(report.get('training_path','/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_tissue_contract_1031_011514/pilot_training4000.jsonl'))
    assert sha(a.pilot/'training_consumed.jsonl')==sha(training)==report['actual_consumed_sha256']
    assert sha(a.pilot/'checkpoint.pt')==report['checkpoint_sha256']
    checkpoint=torch.load(a.pilot/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert int(checkpoint.get('seed',20261031))==int(report.get('seed',20261031))
    head=SharedPromptDecoder(CHANNELS[report['arm']]);head.load_state_dict(checkpoint['model'],strict=True)
    assert all(torch.isfinite(p).all() for p in head.parameters())
    total_updates=1000+report.get('starting_updates',0)
    assert all(int(v['step'])==total_updates for v in checkpoint['optimizer']['state'].values())
    local=LocalData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
    remote=TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    local_plans=[p for p in json.loads((local.tasks/'validation_maximum.json').read_text()) if p['prompt_group']==0]
    wanted=set(json.loads(a.remote_selection.read_text())['pair_ids'])
    remote_plans=[p for p in remote.plans if p['prompt_group']==0 and p['pair_id'] in wanted]
    expected=[]
    for scope,plans in [('local',local_plans),('remote',remote_plans)]:
        for budget in ['3+2','1+1','5+1']:
            for plan in plans:expected.append((scope,budget,plan['query_id'],plan['support_id'],plan['target']))
    actual=[(r['scope'],r['budget'],r['query_id'],r['support_id'],r['target']) for r in rows]
    assert actual==expected and len(rows)==1146
    @lru_cache(maxsize=32)
    def truth(identity):
        r=local.rows[identity]
        with Image.open(r['labels']) as im:gt=np.asarray(im).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest()==r['labels_array_sha256']
        return gt,json.loads(r['input_mpp_xy_json'])
    with np.load(a.pilot/'packed_masks.npz',allow_pickle=False) as f:masks=f['masks'].copy()
    assert masks.shape==(1146,32768)
    for index,row in enumerate(rows):
        assert row['mask_index']==index
        gt,mpp=truth(row['query_id']);pred=np.unpackbits(masks[index],count=512*512).reshape(512,512).astype(bool)
        recounted=score(pred,gt==row['target'],gt!=255,mpp)
        for key,value in recounted.items():
            if value is None:assert row[key] is None
            elif isinstance(value,(int,bool)):assert row[key]==value
            else:assert abs(row[key]-value)<1e-12
    summary={}
    for key,value in report['summary'].items():
        scope,budget=key.split('|');group=[r for r in rows if r['scope']==scope and r['budget']==budget];present=[r for r in group if r['positive']>0];absent=[r for r in group if r['positive']==0]
        dice=float(np.mean([r['dice'] for r in present]));bf1=float(np.mean([r['bf1_um2'] for r in present if r['bf1_um2'] is not None]));fpa=float(np.mean([r['false_positive_fraction'] for r in absent])) if absent else None
        assert abs(dice-value['dice'])<1e-12 and abs(bf1-value['BF1'])<1e-12
        assert (fpa is None and value['absent_FPA'] is None) or abs(fpa-value['absent_FPA'])<1e-12
        summary[key]=dict(dice=dice,BF1=bf1,absent_FPA=fpa,rows=len(group))
    result=dict(status='accepted_fixed_image_pilot',arm=report['arm'],rows_replayed=1146,
        actual4000_task_SHA=report['actual_consumed_sha256'],all1000_optimizer_steps_verified=True,
        all_masks_GT_metrics_replayed=True,summary=summary,total_optimizer_updates=total_updates,
        initial_checkpoint_sha256=report.get('initial_checkpoint_sha256'),start_model_state_sha256=report.get('start_model_state_sha256'),
        qualification='historically exposed fixed-budget development; no image-only versus native SAM or WSI superiority claim')
    (a.pilot/'independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
