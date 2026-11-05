"""Exact continuation gate, fixed 2000-update endpoint, independent mask acceptance."""
import argparse, copy, json, os, time
from pathlib import Path
import numpy as np
import torch
from benchmarks.v5.sam_prior_repair.model import DecoderData, DecoderModel
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans, evaluate_train_monitor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--accept',action='store_true');a=ap.parse_args();r=a.root
    c=json.loads((r/'contract.json').read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for p,h in c['sources'].items():assert sha(p)==h,p
    for p,h in c['inputs'].items():assert sha(p)==h,p
    ref=Path(c['reference']);protocol=Path(c['protocol']);lines=(protocol/'training_consumption.jsonl').read_text().splitlines(keepends=True)
    assert len(lines)==8000 and lines[:4000]==lines[4000:]
    evaluation=[json.loads(x) for x in (protocol/'evaluation.jsonl').read_text().splitlines()];assert len(evaluation)==3711
    data=DecoderData(Path(c['cache']),'sam_native',20261103)
    assert data.identity==json.loads((protocol/'acceptance.json').read_text())['input_identity']
    started=time.monotonic()
    def status(stage,**kw):
        d=dict(stage=stage,worker_pid=os.getpid(),seconds=time.monotonic()-started,**kw);(r/'status.json').write_text(json.dumps(d));print(json.dumps(d),flush=True)
    def save(name,obj): (r/name).write_text(json.dumps(obj,indent=2)+'\n')
    plans=json.loads((r/'monitor_plans.json').read_text())
    assert len(plans)==192
    expected_groups={(d,k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']}
    for offset in range(0,192,32):assert {(p['dataset'],p['kind']) for p in plans[offset:offset+32]}==expected_groups
    if a.accept:
        checkpoint=torch.load(r/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['updates']==2000 and all(int(v['step'])==2000 for v in checkpoint['optimizer']['state'].values())
        assert all(torch.isfinite(x).all() for x in checkpoint['model'].values())
        assert (r/'training_consumed.jsonl').read_text()==''.join(lines)
        rows=json.loads((r/'rows.json').read_text());assert len(rows)==3711
        with np.load(r/'masks.npz') as f:
            for item,row,packed in zip(evaluation,rows,f['masks']):
                assert all(row[k]==item[k] for k in ['scope','budget','plan'])
                gt,mpp=data.labels(item['plan']['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool)
                assert score(mask,gt==item['plan']['target'],gt!=255,mpp)==row['metrics']
        report=json.loads((r/'report.json').read_text());assert report['fresh_replay']==12
        for node in [1000,2000]:
            actual=[]
            for k in range(6):
                monitor=json.loads((r/f'monitor_{node}_{k}.json').read_text());assert monitor['step']==node and len(monitor['rows'])==32
                actual.extend(x['plan'] for x in monitor['rows'])
            assert actual==plans
        guards=compare_rows(json.loads((ref/'rows.json').read_text()),rows)
        result=dict(status='complete_accepted',values={'1000':summarize_arm(json.loads((ref/'rows.json').read_text()))[2],'2000':summarize_arm(rows)[2]},guards=guards,budget_candidate=guards['pass_all'],limitation='one exposed development seed; unequal compute budget diagnosis, not algorithm superiority; no automatic next run',sha256={str(p):sha(p) for p in [r/'checkpoint.pt',r/'rows.json',r/'masks.npz',r/'report.json',r/'contract.json']})
        save('comparison.json',result)
        few=Path(c['fewclick2000']);fewproof=json.loads((few/'comparison.json').read_text())
        for path,digest in fewproof['sha256'].items():assert sha(path)==digest,path
        assert (few/'training_consumed.jsonl').read_text()==''.join(lines)
        fewrows=json.loads((few/'rows.json').read_text())
        fair_guards=compare_rows(rows,fewrows)
        save('fair_comparison.json',dict(status='complete_accepted_equal_update_comparison',
            SAM2000=summarize_arm(rows)[2],FewClick2000=summarize_arm(fewrows)[2],fewclick_vs_sam_guards=fair_guards,
            matched=dict(seed=20261103,updates=2000,consumed_tasks=8000,query_manual_clicks=0,threshold=.5),
            bindings=dict(sam_comparison_sha256=sha(r/'comparison.json'),fewclick_comparison_sha256=sha(few/'comparison.json')),
            limitation='one seed on exposed spatial development set; same task and update budget, different public pretraining/capacity/native resolution/regularization; this SAM bridge is not the upper bound of SAM'))
        status('complete_accepted',budget_candidate=guards['pass_all'],fewclick_vs_sam_guards=fair_guards['pass_all']);return
    proof=json.loads((ref/'independent_acceptance.json').read_text())
    for name,key in [('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256'),('report.json','report_sha256')]:assert sha(ref/name)==proof[key]
    status('resume_gate')
    def build():
        m=DecoderModel('sam_native',20261103,'cuda:0');m.train();o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);return m,o
    def load(m,o,state):m.load_state_dict(state['model']);o.load_state_dict(copy.deepcopy(state['optimizer']));m.train()
    def update(m,o,batch):
        o.zero_grad(set_to_none=True);total=0.
        for task in batch:
            assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
            inputs=data.model_inputs(task,'xcell','cuda:0');loss=episode_loss(m(**inputs),data,task,inputs['prompts']);assert torch.isfinite(loss);(loss/4).backward();total+=float(loss.detach())/4
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);o.step();return total
    state=torch.load(ref/'checkpoint.pt',map_location='cuda:0',weights_only=True);assert state['updates']==1000
    m,o=build();load(m,o,state)
    # Prove training has no stochastic operations: missing historical RNG then cannot affect continuation.
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    update(m,o,[json.loads(x) for x in lines[4000:4004]])
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()),'Historical RNG missing and stochastic training detected: stop'
    expected=(tree_sha(m.state_dict()),tree_sha(o.state_dict()))
    torch.save(dict(model=m.state_dict(),optimizer=o.state_dict()),r/'gate_step1001.pt')
    update(m,o,[json.loads(x) for x in lines[4004:4008]]);expected2=(tree_sha(m.state_dict()),tree_sha(o.state_dict()))
    del m,o;torch.cuda.empty_cache();m,o=build();load(m,o,state)
    update(m,o,[json.loads(x) for x in lines[4000:4004]]);assert expected==(tree_sha(m.state_dict()),tree_sha(o.state_dict()))
    load(m,o,torch.load(r/'gate_step1001.pt',map_location='cuda:0',weights_only=True));update(m,o,[json.loads(x) for x in lines[4004:4008]]);assert expected2==(tree_sha(m.state_dict()),tree_sha(o.state_dict()))
    load(m,o,state);del state
    # Baseline prediction must reproduce the accepted reference before extending training.
    m.eval();old=json.loads((ref/'rows.json').read_text())
    with np.load(ref/'masks.npz') as f,torch.no_grad():
        for index in range(12):
            item=evaluation[index];logits=m(**data.model_inputs(item['plan'],'xcell','cuda:0'))['query']['pixel_logits'][0]
            assert np.array_equal(np.packbits((logits>=0).cpu().numpy()),f['masks'][index])
    save('resume_acceptance.json',dict(status='accepted_exact_continuation',optimizer_step1001_exact=True,disk_step1002_exact=True,training_rng_unchanged=True,reference_masks_replayed=12))
    def monitor(node):
        status('train_monitor',updates=node)
        # Six groups of 32 tasks, scoring GT accessed after inference only.
        for k in range(6):
            evaluate_train_monitor(m,data,plans[k*32:(k+1)*32],'cuda:0',node,r/f'monitor_{node}_{k}.json')
            status('train_monitor',updates=node,completed_tasks=(k+1)*32,total_tasks=192)
        source=Path(c['training_source']);evaluate_train_monitor(m,data,fixed_train_plans(source),'cuda:0',node,r/f'legacy_monitor_{node}.json')
    monitor(1000);m.train()
    backbone=tree_sha({k:v for k,v in m.state_dict().items() if '.image_encoder.' in k})
    with (r/'training_consumed.jsonl').open('w') as stream:
        stream.write(''.join(lines[:4000]))
        for step in range(1000,2000):
            loss=update(m,o,[json.loads(x) for x in lines[step*4:step*4+4]]);stream.write(''.join(lines[step*4:step*4+4]))
            if step%25==0 or step==1999:stream.flush();status('training',updates=step+1,total_updates=2000,loss=loss)
    assert backbone==tree_sha({k:v for k,v in m.state_dict().items() if '.image_encoder.' in k})
    torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),updates=2000,seed=20261103,arm='sam_native',cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),r/'checkpoint.pt')
    monitor(2000);m.eval();rows=[];masks=[];replay=[]
    with torch.no_grad():
        for item in evaluation:
            logits=m(**data.model_inputs(item['plan'],'xcell','cuda:0'))['query']['pixel_logits'];assert torch.isfinite(logits).all()
            mask=(logits[0]>=0).cpu().numpy();gt,mpp=data.labels(item['plan']['query_id']);rows.append(item|dict(metrics=score(mask,gt==item['plan']['target'],gt!=255,mpp)));masks.append(np.packbits(mask))
            if len(replay)<12:replay.append(logits.cpu())
            if len(rows)%25==0:status('scoring',rows=len(rows),total_rows=len(evaluation))
    del m,o;torch.cuda.empty_cache();m,_=build();m.load_state_dict(torch.load(r/'checkpoint.pt',map_location='cuda:0',weights_only=True)['model']);m.eval()
    with torch.no_grad():
        for item,expected in zip(evaluation,replay):assert torch.equal(m(**data.model_inputs(item['plan'],'xcell','cuda:0'))['query']['pixel_logits'].cpu(),expected)
    save('rows.json',rows);np.savez_compressed(r/'masks.npz',masks=np.stack(masks));save('report.json',dict(updates=2000,added_updates=1000,fresh_replay=12,seconds=time.monotonic()-started,contract_sha256=sha(r/'contract.json')));status('complete_unaccepted')

if __name__=='__main__':main()
