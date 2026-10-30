"""Fixed1000x4 prompt-mask pilot and paired local/remote endpoint scoring."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from .image_features import CHANNELS
from .shared_prompt import SharedPromptDecoder
from .gate_shared_decoder import loss_fn
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.transfer_comparison.data import TransferData


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--remote-selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    accepted=json.loads((a.cache/'acceptance.json').read_text());assert accepted['status']=='accepted_all1737_spatial_features'
    arm=accepted['arm'];entries={r['tile_id']:r for r in [json.loads(s) for s in (a.cache/'index.jsonl').read_text().splitlines()]}
    local=LocalData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
    remote=TransferData('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')
    training_path=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_tissue_contract_1031_011514/pilot_training4000.jsonl')
    lines=training_path.read_text().splitlines(keepends=True);assert len(lines)==4000
    status=dict(status='training',worker_pid=os.getpid(),arm=arm,updates=0,tasks=0)
    def save():
        status['elapsed_seconds']=time.time()-start
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    @lru_cache(maxsize=32)
    def inputs(identity):
        row=entries[identity];feature=np.load(row['feature'],allow_pickle=False)
        assert hashlib.sha256(feature.tobytes()).hexdigest()==row['array_sha256'] and sha(row['image'])==row['image_sha256']
        with Image.open(row['image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
        return torch.from_numpy(feature.astype('float32'))[None],torch.from_numpy(rgb).permute(2,0,1)[None].float()/255
    @lru_cache(maxsize=32)
    def labels(identity):
        row=local.rows[identity]
        with Image.open(row['labels']) as im:gt=np.asarray(im).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest()==row['labels_array_sha256']
        return gt,np.array(json.loads(row['input_mpp_xy_json']))
    def predict(plan,model):
        qf,qr=(v.to('cuda:0') for v in inputs(plan['query_id']))
        sf=qf if plan['support_id']==plan['query_id'] else inputs(plan['support_id'])[0].to('cuda:0')
        positive=torch.tensor([plan['positive_xy']],device='cuda:0',dtype=torch.float32)
        negative=torch.tensor([plan['negative_xy']],device='cuda:0',dtype=torch.float32)
        return model(qf,qr,positive,negative,support_feature=sf,local=plan['support_id']==plan['query_id'])
    torch.manual_seed(20261031);model=SharedPromptDecoder(CHANNELS[arm]).to('cuda:0')
    common_sha=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for k,v in sorted(model.state_dict().items()) if not k.startswith('project.'))).hexdigest()
    assert common_sha=='552f5609885b2ea3f6545f537f7bd01aa1c5b43cda66405545a41e681b1bb4c4'
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);save();losses=[]
    with (a.output/'training_consumed.jsonl').open('w') as stream:
        for step in range(1000):
            optimizer.zero_grad();total=0.
            for line in lines[step*4:(step+1)*4]:
                plan=json.loads(line);assert plan['split']=='train' and plan['kind']=='local'
                logits=predict(plan,model);gt,_=labels(plan['query_id']);valid=torch.from_numpy(gt!=255)[None].to('cuda:0')
                truth=torch.from_numpy(gt==plan['target'])[None].to('cuda:0')
                loss=loss_fn(logits,truth,valid)/4;assert torch.isfinite(loss);loss.backward();total+=float(loss.detach());stream.write(line)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();losses.append(total)
            if step%25==0:stream.flush();status.update(updates=step+1,tasks=(step+1)*4,loss=total);save();print(json.dumps(status),flush=True)
    assert sha(a.output/'training_consumed.jsonl')==sha(training_path)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':1000},a.output/'checkpoint.pt')
    sample=json.loads(lines[0]);model.eval()
    with torch.no_grad():reference=predict(sample,model)
    fresh=SharedPromptDecoder(CHANNELS[arm]).to('cuda:0');checkpoint=torch.load(a.output/'checkpoint.pt',map_location='cuda:0',weights_only=True);fresh.load_state_dict(checkpoint['model'],strict=True);fresh.eval()
    with torch.no_grad():assert torch.equal(reference,predict(sample,fresh))
    selection=json.loads(a.remote_selection.read_text());wanted=set(selection['pair_ids'])
    remote_plans=[p for p in remote.plans if p['prompt_group']==0 and p['pair_id'] in wanted]
    assert len(remote_plans)==len(wanted)==80
    rows=[];packed=[];status.update(status='scoring',updates=1000,tasks=4000);save()
    with torch.inference_mode():
        for scope,plans in [('local',[p for p in json.loads((local.tasks/'validation_maximum.json').read_text()) if p['prompt_group']==0]),('remote',remote_plans)]:
            for budget in ['3+2','1+1','5+1']:
                for original in plans:
                    plan=nested(original,budget);logits=predict(plan,fresh);assert torch.isfinite(logits).all()
                    pred=(logits[0]>=0).cpu().numpy()
                    # Query GT is first accessed only after inference for this evaluation row.
                    gt,mpp=labels(plan['query_id']);metrics=score(pred,gt==plan['target'],gt!=255,mpp)
                    row=dict(scope=scope,budget=budget,query_id=plan['query_id'],support_id=plan['support_id'],
                        target=plan['target'],dataset=plan['dataset'],kind=plan['kind'],mask_index=len(packed),**metrics)
                    rows.append(row);packed.append(np.packbits(pred.reshape(-1)))
                status.update(scoring_rows=len(rows));save()
    np.savez_compressed(a.output/'packed_masks.npz',masks=np.stack(packed))
    (a.output/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary={}
    for scope in ['local','remote']:
        for budget in ['3+2','1+1','5+1']:
            group=[r for r in rows if r['scope']==scope and r['budget']==budget]
            present=[r for r in group if r['positive']>0];absent=[r for r in group if r['positive']==0]
            summary[scope+'|'+budget]=dict(rows=len(group),present=len(present),absent=len(absent),
                dice=float(np.mean([r['dice'] for r in present])),BF1=float(np.mean([r['bf1_um2'] for r in present if r['bf1_um2'] is not None])),
                absent_FPA=float(np.mean([r['false_positive_fraction'] for r in absent])) if absent else None)
    report=dict(status='complete_shared_image_pilot_unaccepted',worker_pid=os.getpid(),arm=arm,summary=summary,
        training_updates=1000,training_tasks=4000,actual_consumed_sha256=sha(a.output/'training_consumed.jsonl'),
        common_initialization_sha256=common_sha,exact_final_reload=True,checkpoint_sha256=sha(a.output/'checkpoint.pt'),
        cache_index_sha256=accepted['index_sha256'],remote_selection_sha256=sha(a.remote_selection),source_sha256=sha(__file__),
        elapsed_seconds=time.time()-start,losses=losses,qualification='fixed-budget frozen encoder development pilot, not native SAM system or final WSI superiority')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete',scoring_rows=len(rows));save();print(json.dumps(dict(arm=arm,summary=summary)),flush=True)


if __name__=='__main__':main()
