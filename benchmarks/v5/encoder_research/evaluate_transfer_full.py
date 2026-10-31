"""Fixed-checkpoint complete frozen935 transfer cohort; no training/selection."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from .shared_prompt import SharedPromptDecoder
from .image_features import CHANNELS
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.transfer_comparison.data import TransferData


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--reference-pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reference=json.loads((a.reference_pilot/'report.json').read_text())
    accepted=json.loads((a.reference_pilot/'independent_acceptance.json').read_text());assert accepted['status']=='accepted_fixed_image_pilot'
    cache=json.loads((a.cache/'acceptance.json').read_text());assert cache['status']=='accepted_all1737_spatial_features'
    assert cache['index_sha256']==reference['cache_index_sha256']
    checkpoint=a.reference_pilot/'checkpoint.pt';assert sha(checkpoint)==reference['checkpoint_sha256']
    model=SharedPromptDecoder(CHANNELS[reference['arm']]);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True);model.to('cuda:0').eval().requires_grad_(False)
    data=TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    plans=[p for p in data.plans if p['prompt_group']==0];assert len(plans)==935
    entries={r['tile_id']:r for r in [json.loads(s) for s in (a.cache/'index.jsonl').read_text().splitlines()]}
    old_rows=json.loads((a.reference_pilot/'rows.json').read_text())
    with np.load(a.reference_pilot/'packed_masks.npz',allow_pickle=False) as f:old_masks=f['masks'].copy()
    old={(r['budget'],r['support_id'],r['query_id'],r['target']):old_masks[r['mask_index']] for r in old_rows if r['scope']=='remote'}
    assert len(old)==240
    if a.gate:
        selected={}
        for plan in plans:
            key=('3+2',plan['support_id'],plan['query_id'],plan['target'])
            if key in old:selected.setdefault((plan['dataset'],plan['kind']),plan)
        plans=list(selected.values());assert len(plans)==8
    status=dict(status='scoring',worker_pid=os.getpid(),completed=0,total=len(plans)*(1 if a.gate else 3))
    def save():
        status['elapsed_seconds']=time.time()-start
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    @lru_cache(maxsize=64)
    def inputs(identity):
        row=entries[identity];feature=np.load(row['feature'],allow_pickle=False)
        assert hashlib.sha256(feature.tobytes()).hexdigest()==row['array_sha256'] and sha(row['image'])==row['image_sha256']
        with Image.open(row['image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        return torch.from_numpy(feature.astype('float32'))[None],torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
    save();results=[];masks=[];overlap_rows=0
    with torch.inference_mode():
        for budget in (['3+2'] if a.gate else ['3+2','1+1','5+1']):
            for original in plans:
                plan=nested(original,budget);qf,qr=(x.to('cuda:0') for x in inputs(plan['query_id']));sf=inputs(plan['support_id'])[0].to('cuda:0')
                positive=torch.tensor([plan['positive_xy']],device='cuda:0',dtype=torch.float32);negative=torch.tensor([plan['negative_xy']],device='cuda:0',dtype=torch.float32)
                logits=model(qf,qr,positive,negative,support_feature=sf,local=False);assert torch.isfinite(logits).all()
                pred=(logits[0]>=0).cpu().numpy();packed=np.packbits(pred.reshape(-1))
                key=(budget,plan['support_id'],plan['query_id'],plan['target'])
                if key in old:assert np.array_equal(packed,old[key]),'fixed checkpoint80-subset reproduction failed';overlap_rows+=1
                # Scoring labels are accessed after prediction; no GT/candidate choice in forward.
                gt=data.labels(plan['query_id']);valid=gt!=255;mpp=json.loads(data.rows[plan['query_id']]['input_mpp_xy_json'])
                metrics=score(pred,gt==plan['target'],valid,mpp)
                results.append(dict(pair_id=plan['pair_id'],dataset=plan['dataset'],target=plan['target'],kind=plan['kind'],
                    support_id=plan['support_id'],query_id=plan['query_id'],budget=budget,mask_index=len(masks),**metrics))
                masks.append(packed)
                if len(results)%25==0:status['completed']=len(results);save()
    assert overlap_rows==(8 if a.gate else 240)
    np.savez_compressed(a.output/'packed_masks.npz',masks=np.stack(masks))
    (a.output/'rows.json').write_text(json.dumps(results,indent=2)+'\n')
    report=dict(status='complete_transfer_input_gate' if a.gate else 'complete_frozen935_transfer_unaccepted',worker_pid=os.getpid(),
        rows=len(results),reference_subset_rows_reproduced=overlap_rows,exact_reference_mask_equality=True,
        reference_pilot=str(a.reference_pilot),checkpoint_sha256=sha(checkpoint),cache_index_sha256=cache['index_sha256'],
        frozen_tasks_sha256=sha(data.tasks/'validation_maximum.json'),source_sha256=sha(__file__),
        elapsed_seconds=time.time()-start,training_steps=0,threshold=.5,scope='full previously frozen development transfer cohort; not independent blind test')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete',completed=len(results));save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
