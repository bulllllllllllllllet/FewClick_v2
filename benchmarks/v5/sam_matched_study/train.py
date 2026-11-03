# Source-preserving adaptation of train_spatial: matched decoder factory and per-arm throughput.
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
from .model import DecoderData as SpatialData,DecoderModel as SpatialModel,ARMS
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans,evaluate_train_monitor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--learning',type=Path,required=True);p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--accept',action='store_true')
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--device',default='cuda:0')
    p.add_argument('--execution-contract',type=Path);p.add_argument('--benchmark-receipt',type=Path);p.add_argument('--monitor-receipt',type=Path);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((a.protocol/'acceptance.json').read_text());assert protocol['status']=='accepted_spatial_task_protocol_only'
    assert sha(a.protocol/'training_consumption.jsonl')==protocol['consumption_sha256'] and sha(a.protocol/'evaluation.jsonl')==protocol['evaluation_sha256']
    data=SpatialData(a.cache,a.arm,a.seed);assert data.identity==protocol['input_identity']
    lines=(a.protocol/'training_consumption.jsonl').read_text().splitlines(keepends=True);assert len(lines)==8000
    evaluation=list(map(json.loads,(a.protocol/'evaluation.jsonl').read_text().splitlines()));assert len(evaluation)==3711
    updates=2 if a.smoke else 1000
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
        nodes=[0] if a.smoke else [0,250,500,1000]
        assert report['train_monitor_nodes']==nodes
        for node in nodes:
            monitor=a.output/f'train_monitor_{node}.json';assert sha(monitor)==report['train_monitor_sha256'][str(node)]
            item=json.loads(monitor.read_text());assert len(item['rows'])==32 and item['step']==node and not item['selection_used']
        (a.output/'independent_acceptance.json').write_text(json.dumps(dict(status='accepted_P3_spatial_evaluation',arm=a.arm,seed=a.seed,smoke=a.smoke,
             report_sha256=sha(a.output/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],rows=len(rows),rows_sha256=sha(a.output/'rows.json'),masks_sha256=sha(a.output/'masks.npz')),indent=2)+'\n');return
    for arm in ([a.arm] if a.smoke else ARMS):
        receipt=json.loads((a.learning/arm/'independent_acceptance.json').read_text());assert receipt['status']=='accepted_P3_spatial_train_learning_gate' and receipt['arm']==arm
        assert receipt['report_sha256']==sha(a.learning/arm/'report.json')
    if not a.smoke:
        assert a.device.startswith('cuda') and a.inventory
        assert json.loads(a.inventory.read_text())['status']=='accepted_full_spatial_inventory'
        assert a.execution_contract and a.benchmark_receipt and a.monitor_receipt,'Formal run needs frozen execution and actual probe/monitor acceptance'
        execution=json.loads(a.execution_contract.read_text());assert execution['updates']==1000 and execution['batch']==4 and execution['seed']==a.seed
        assert execution['protocol_sha256']==sha(a.protocol/'acceptance.json')
        for path,digest in execution['sources'].items():assert sha(path)==digest,path
        assert execution['sources'][str(Path(__file__).relative_to(Path.cwd()))]==sha(__file__)
        benchmark=json.loads(a.benchmark_receipt.read_text());assert benchmark['status']=='accepted_support_match_GPU_probe' and benchmark['arm']==a.arm and benchmark['updates']==20 and benchmark['inference_tasks']==100
        assert json.loads(a.monitor_receipt.read_text())['status']=='accepted_fixed_train_monitor_integration'
        # Explicitly require a separate accepted smoke on this arm before formal budget.
        smoke=json.loads((a.output.parent/(a.arm+'_smoke')/'independent_acceptance.json').read_text());assert smoke['smoke'] and smoke['arm']==a.arm
    a.output.mkdir(exist_ok=False);model=SpatialModel(a.arm,a.seed,a.device);model.train()
    initial_sha=tree_sha(model.state_dict());common_sha=tree_sha(model.base.state_dict());backbone_sha=tree_sha({k:v for k,v in model.state_dict().items() if '.image_encoder.' in k})
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);started=time.monotonic()
    def status(**value):
        value.update(worker_pid=os.getpid(),arm=a.arm,seconds=time.monotonic()-started)
        (a.output/'status.json').write_text(json.dumps(value));print(json.dumps(value),flush=True)
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
    monitor_plans=fixed_train_plans(source);monitors={}
    def monitor(node):
        status(stage='train_monitor',node=node,tasks=32)
        path=a.output/f'train_monitor_{node}.json'
        evaluate_train_monitor(model,data,monitor_plans,a.device,node,path);monitors[str(node)]=sha(path)
        status(stage='train_monitor_complete',node=node,tasks=32)
    monitor(0)
    with (a.output/'training_consumed.jsonl').open('w') as stream:
        for step in range(updates):
            opt.zero_grad(set_to_none=True);loss_sum=0.
            for line in lines[step*4:step*4+4]:
                task=json.loads(line);assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
                inputs=data.model_inputs(task,'xcell',a.device);loss=episode_loss(model(**inputs),data,task,inputs['prompts'])
                assert torch.isfinite(loss);(loss/4).backward();loss_sum+=float(loss.detach())/4;stream.write(line)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step()
            if step%25==0 or step==updates-1:stream.flush();status(stage='training',updates=step+1,total_updates=updates,loss=loss_sum)
            if step+1 in [250,500,1000]:monitor(step+1)
    assert tree_sha({k:v for k,v in model.state_dict().items() if '.image_encoder.' in k})==backbone_sha
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=a.arm,seed=a.seed,updates=updates),a.output/'checkpoint.pt')
    model.eval();rows=[];masks=[];references=[]
    last_progress=time.monotonic()
    with torch.no_grad():
        for item in evaluation:
            task=item['plan'];logits=model(**data.model_inputs(task,'xcell',a.device))['query']['pixel_logits'];assert torch.isfinite(logits).all()
            mask=(logits[0]>=0).cpu().numpy();gt,mpp=data.labels(task['query_id']);rows.append(item|dict(metrics=score(mask,gt==task['target'],gt!=255,mpp)));masks.append(np.packbits(mask))
            if len(references)<12:references.append(logits.cpu())
            if len(rows)==1 or len(rows)%25==0 or len(rows)==len(evaluation) or time.monotonic()-last_progress>=30:
                status(stage='scoring',rows=len(rows),total_rows=len(evaluation));last_progress=time.monotonic()
    del model;gc.collect()
    if a.device.startswith('cuda'):torch.cuda.empty_cache()
    fresh=SpatialModel(a.arm,a.seed,a.device);fresh.load_state_dict(torch.load(a.output/'checkpoint.pt',weights_only=True,map_location=a.device)['model']);fresh.eval()
    with torch.no_grad():
        for index,(item,ref) in enumerate(zip(evaluation,references)):
            assert torch.equal(fresh(**data.model_inputs(item['plan'],'xcell',a.device))['query']['pixel_logits'].cpu(),ref)
            status(stage='fresh_prediction_replay',rows=index+1,total_rows=len(references))
    np.savez_compressed(a.output/'masks.npz',masks=np.stack(masks));(a.output/'rows.json').write_text(json.dumps(rows)+'\n')
    report=dict(arm=a.arm,seed=a.seed,smoke=a.smoke,device=a.device,updates=updates,consumed_tasks=updates*4,inputs=data.identity,
                initial_state_sha256=initial_sha,initial_common_sha256=common_sha,source_sha256=sha(__file__),protocol_sha256=sha(a.protocol/'acceptance.json'),
                checkpoint_sha256=sha(a.output/'checkpoint.pt'),fresh_prediction_replay_count=len(references),seconds=time.monotonic()-started,
                train_monitor_nodes=[int(x) for x in monitors],train_monitor_sha256=monitors,
                execution_contract_sha256=sha(a.execution_contract) if a.execution_contract else None)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status(stage='complete_unaccepted',rows=len(rows))


if __name__=='__main__':main()
