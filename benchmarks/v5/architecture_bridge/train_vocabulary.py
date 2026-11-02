"""Fixed-budget four-arm original-architecture bridge, with full mask replay."""
import argparse,json,os,time
from pathlib import Path
import pyvips
import numpy as np
import torch
from .data import BridgeData,BASE
from .restored_loss import episode_loss
from .vocabulary_model import VocabularyFewClick
from .gate_vocabulary_model import common_state
ARMS={m:('phikon','xcell') for m in ['continuous','fixed','ema']}
from .gate_image_zero_head import heads
from benchmarks.v5.encoder_research.continuous_data import ContinuousData,sha
from benchmarks.v5.encoder_research.continuous_prompt import mask_loss,tree_sha
from benchmarks.v5.fair_comparison.metrics import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--accept',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=BridgeData();source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';lines=source.read_text().splitlines(keepends=True);assert len(lines)==4000
    evaluation=list(ContinuousData.validation(data));assert len(evaluation)==3711
    updates=2 if a.smoke else 2000
    training_lines=lines*2
    if a.smoke:
        seen=set();chosen=[]
        for item in evaluation:
            key=(item[0],item[1],item[2]['dataset'])
            if key not in seen: seen.add(key);chosen.append(item)
        evaluation=chosen;assert len(evaluation)==12
    if a.accept:
        report=json.loads((a.output/'report.json').read_text());assert report['arm']==a.arm and report['updates']==updates
        assert report['checkpoint_sha256']==sha(a.output/'checkpoint.pt') and report['training_sha256']==sha(source)
        assert (a.output/'training_consumed.jsonl').read_text()==''.join(training_lines[:updates*4])
        rows=json.loads((a.output/'rows.json').read_text());masks=np.load(a.output/'masks.npz',allow_pickle=False)['masks'];assert len(rows)==len(masks)==len(evaluation)
        for i,(scope,budget,plan) in enumerate(evaluation):
            row=rows[i];assert row['scope']==scope and row['budget']==budget and row['plan']==plan
            gt,mpp=data.labels(plan['query_id']);mask=np.unpackbits(masks[i],count=512*512).reshape(512,512).astype(bool)
            assert score(mask,gt==plan['target'],gt!=255,mpp)==row['metrics']
        (a.output/'independent_acceptance.json').write_text(json.dumps(dict(status='accepted_P2_vocabulary_evaluation',arm=a.arm,worker_pid=os.getpid(),rows=len(rows),smoke=a.smoke,report_sha256=sha(a.output/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],quality_compared=False),indent=2)+'\n');return
    gate=json.loads((a.gate/'independent_gate_acceptance.json').read_text());assert gate['status']=='accepted_all_three_vocabulary_gates'
    receipts=[json.loads((a.gate/arm/'report.json').read_text()) for arm in ARMS]
    assert len({r['initial_common_heads_sha256'] for r in receipts})==1
    for r in receipts:
        assert r['exact_optimizer_continuation'] and r['inputs']==data.identity
        for name,value in r['sources'].items():assert sha(Path(__file__).with_name(name))==value
    a.output.mkdir(exist_ok=False);image_arm,cell_arm=ARMS[a.arm];model=VocabularyFewClick(a.arm);identity=model.identity;model.train()
    assert tree_sha(common_state(model))==receipts[0]['initial_common_heads_sha256']
    initial_backbone=tree_sha({k:v for k,v in model.state_dict().items() if '.backbone.' in k})
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);started=time.time()
    with (a.output/'training_consumed.jsonl').open('w') as stream:
        for step in range(updates):
            opt.zero_grad(set_to_none=True);batch=training_lines[step*4:step*4+4];total=0.
            for line in batch:
                plan=json.loads(line);assert data.rows[plan['query_id']]['split']==data.rows[plan['support_id']]['split']=='train'
                output=model(**data.model_inputs(plan,cell_arm,'cuda:0'));gt,_=data.labels(plan['query_id'])
                loss=episode_loss(output,data,plan,data.model_inputs(plan,cell_arm,'cuda:0')['prompts'])
                assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4;stream.write(line)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step();model.after_optimizer_step()
            if step%25==0 or step==updates-1:
                stream.flush();status=dict(status='training',arm=a.arm,worker_pid=os.getpid(),updates=step+1,total_updates=updates,loss=total,seconds=time.time()-started)
                (a.output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    assert tree_sha({k:v for k,v in model.state_dict().items() if '.backbone.' in k})==initial_backbone
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=updates,arm=a.arm,identity=identity),a.output/'checkpoint.pt')
    saved=torch.load(a.output/'checkpoint.pt',map_location='cpu',weights_only=True);assert tree_sha(saved['model'])==tree_sha(model.state_dict())
    model.eval();rows=[];packed=[]
    with torch.no_grad():
        for scope,budget,plan in evaluation:
            logits=model(**data.model_inputs(plan,cell_arm,'cuda:0'))['query']['pixel_logits'];assert torch.isfinite(logits).all()
            mask=(logits[0]>=0).cpu().numpy();gt,mpp=data.labels(plan['query_id'])
            rows.append(dict(scope=scope,budget=budget,plan=plan,metrics=score(mask,gt==plan['target'],gt!=255,mpp)));packed.append(np.packbits(mask))
            if len(rows)%100==0:print(json.dumps(dict(status='scoring',rows=len(rows))),flush=True)
    np.savez_compressed(a.output/'masks.npz',masks=np.stack(packed));(a.output/'rows.json').write_text(json.dumps(rows)+'\n')
    report=dict(arm=a.arm,worker_pid=os.getpid(),updates=updates,consumed_tasks=updates*4,unique_source_tasks=4000,epochs=2 if not a.smoke else None,smoke=a.smoke,identity=identity,inputs=data.identity,training_sha256=sha(source),checkpoint_sha256=sha(a.output/'checkpoint.pt'),gate_sha256=sha(a.gate/'independent_gate_acceptance.json'),source_sha256=sha(__file__),seconds=time.time()-started)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
