"""P3 fixed-consumption trainer, final-checkpoint evaluation and mask acceptance."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import pyvips
import numpy as np
import torch
from .spatial_controls import SpatialData,SpatialModel,ARMS
from .restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--learning',type=Path,required=True);p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--accept',action='store_true')
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((a.protocol/'acceptance.json').read_text());assert protocol['status']=='accepted_spatial_task_protocol_only'
    assert sha(a.protocol/'training_consumption.jsonl')==protocol['consumption_sha256'] and sha(a.protocol/'evaluation.jsonl')==protocol['evaluation_sha256']
    data=SpatialData(a.cache,a.arm,a.seed);assert data.identity==protocol['input_identity']
    lines=(a.protocol/'training_consumption.jsonl').read_text().splitlines(keepends=True);assert len(lines)==8000
    evaluation=list(map(json.loads,(a.protocol/'evaluation.jsonl').read_text().splitlines()));assert len(evaluation)==3711
    updates=2 if a.smoke else 2000
    if a.smoke:
        seen=set();chosen=[]
        for item in evaluation:
            key=(item['scope'],item['budget'],item['plan']['dataset'])
            if key not in seen:chosen.append(item);seen.add(key)
        evaluation=chosen;assert len(evaluation)==12
    if a.accept:
        report=json.loads((a.output/'report.json').read_text());assert report['arm']==a.arm and report['seed']==a.seed and report['smoke']==a.smoke
        assert report['updates']==updates and report['checkpoint_sha256']==sha(a.output/'checkpoint.pt') and report['protocol_sha256']==sha(a.protocol/'acceptance.json')
        assert (a.output/'training_consumed.jsonl').read_text()==''.join(lines[:updates*4])
        rows=json.loads((a.output/'rows.json').read_text())
        with np.load(a.output/'masks.npz') as f:masks=f['masks']
        assert len(rows)==len(masks)==len(evaluation)
        for row,item,packed in zip(rows,evaluation,masks):
            assert {k:row[k] for k in ['scope','budget','plan']}==item
            gt,mpp=data.labels(item['plan']['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool)
            assert score(mask,gt==item['plan']['target'],gt!=255,mpp)==row['metrics']
        assert report['fresh_prediction_replay_count']==min(12,len(evaluation))
        (a.output/'independent_acceptance.json').write_text(json.dumps(dict(status='accepted_P3_spatial_evaluation',arm=a.arm,seed=a.seed,smoke=a.smoke,
             report_sha256=sha(a.output/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],rows=len(rows),rows_sha256=sha(a.output/'rows.json'),masks_sha256=sha(a.output/'masks.npz')),indent=2)+'\n');return
    for arm in ([a.arm] if a.smoke else ARMS):
        receipt=json.loads((a.learning/arm/'independent_acceptance.json').read_text());assert receipt['status']=='accepted_P3_spatial_train_learning_gate' and receipt['arm']==arm
        assert receipt['report_sha256']==sha(a.learning/arm/'report.json')
    if not a.smoke:
        assert a.device.startswith('cuda') and a.inventory
        assert json.loads(a.inventory.read_text())['status']=='accepted_full_spatial_inventory'
        # Explicitly require a separate accepted smoke on this arm before formal budget.
        smoke=json.loads((a.output.parent/(a.arm+'_smoke')/'independent_acceptance.json').read_text());assert smoke['smoke'] and smoke['arm']==a.arm
    a.output.mkdir(exist_ok=False);model=SpatialModel(a.arm,a.seed,a.device);model.train()
    initial_sha=tree_sha(model.state_dict());backbone_sha=tree_sha({k:v for k,v in model.state_dict().items() if '.backbone.' in k})
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);started=time.monotonic()
    def status(**value):
        value.update(worker_pid=os.getpid(),arm=a.arm,seconds=time.monotonic()-started)
        (a.output/'status.json').write_text(json.dumps(value));print(json.dumps(value),flush=True)
    with (a.output/'training_consumed.jsonl').open('w') as stream:
        for step in range(updates):
            opt.zero_grad(set_to_none=True);loss_sum=0.
            for line in lines[step*4:step*4+4]:
                task=json.loads(line);assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
                inputs=data.model_inputs(task,'xcell',a.device);loss=episode_loss(model(**inputs),data,task,inputs['prompts'])
                assert torch.isfinite(loss);(loss/4).backward();loss_sum+=float(loss.detach())/4;stream.write(line)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
            if step%25==0 or step==updates-1:stream.flush();status(stage='training',updates=step+1,total_updates=updates,loss=loss_sum)
    assert tree_sha({k:v for k,v in model.state_dict().items() if '.backbone.' in k})==backbone_sha
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=a.arm,seed=a.seed,updates=updates),a.output/'checkpoint.pt')
    model.eval();rows=[];masks=[];references=[]
    with torch.no_grad():
        for item in evaluation:
            task=item['plan'];logits=model(**data.model_inputs(task,'xcell',a.device))['query']['pixel_logits'];assert torch.isfinite(logits).all()
            mask=(logits[0]>=0).cpu().numpy();gt,mpp=data.labels(task['query_id']);rows.append(item|dict(metrics=score(mask,gt==task['target'],gt!=255,mpp)));masks.append(np.packbits(mask))
            if len(references)<12:references.append(logits.cpu())
            if len(rows)%100==0:status(stage='scoring',rows=len(rows),total_rows=len(evaluation))
    del model;gc.collect()
    if a.device.startswith('cuda'):torch.cuda.empty_cache()
    fresh=SpatialModel(a.arm,a.seed,a.device);fresh.load_state_dict(torch.load(a.output/'checkpoint.pt',weights_only=True,map_location=a.device)['model']);fresh.eval()
    with torch.no_grad():
        for item,ref in zip(evaluation,references):assert torch.equal(fresh(**data.model_inputs(item['plan'],'xcell',a.device))['query']['pixel_logits'].cpu(),ref)
    np.savez_compressed(a.output/'masks.npz',masks=np.stack(masks));(a.output/'rows.json').write_text(json.dumps(rows)+'\n')
    report=dict(arm=a.arm,seed=a.seed,smoke=a.smoke,device=a.device,updates=updates,consumed_tasks=updates*4,inputs=data.identity,
                initial_state_sha256=initial_sha,source_sha256=sha(__file__),protocol_sha256=sha(a.protocol/'acceptance.json'),
                checkpoint_sha256=sha(a.output/'checkpoint.pt'),fresh_prediction_replay_count=len(references),seconds=time.monotonic()-started)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(stage='complete_unaccepted',rows=len(rows))


if __name__=='__main__':main()
