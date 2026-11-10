from .model import frozen_backbone_state
# Adapted from architecture_bridge/functional_repaired.py: only model/data factories change.
"""Fixed train-only local/remote/absent learning gate with fresh final replay."""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.v5.architecture_bridge.data import BASE
from benchmarks.v5.architecture_bridge.repaired_model import RepairedData, RepairedContinuous
from .loss import episode_loss
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score

BOUNDS=dict(present_group_Dice_min=.2,absent_group_FPA_max=.1,
            present_group_mean_std_min=.02,present_group_median_swap_mean_min=.001,
            region_present_group_Dice_min=.15,region_absent_group_FPA_max=.1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['aux_original','aux_detail'],required=True);p.add_argument('--seed',type=int,default=20261103);a=p.parse_args()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(exist_ok=False)
    source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl'
    plans=list(map(json.loads,source.read_text().splitlines()));selected=[]
    for d in ['crc','tiger']:
        for kind,n in [('local',8),('cross_present',4),('cross_absent',4)]:
            group=[r for r in plans if r['dataset']==d and r['kind']==kind][:n];assert len(group)==n;selected+=group
    if a.arm:
        from .model import DecoderData as SpatialData,DecoderModel as SpatialModel
        data=SpatialData(a.cache,a.arm,a.seed)
        build=lambda:SpatialModel(a.arm,a.seed)
    else:
        data=RepairedData(a.cache)
        build=lambda:RepairedContinuous(a.seed)
    assert all(data.rows[r['support_id']]['split']==data.rows[r['query_id']]['split']=='train' for r in selected)
    (a.output/'contract.json').write_text(json.dumps(dict(tasks=selected,source_sha256=sha(source),cache=data.identity,bounds=BOUNDS,updates=256,batch=4,seed=a.seed,arm=a.arm),indent=2))
    started=time.monotonic()
    model=build();model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
    initial_trainable_sha=tree_sha({n:p.detach() for n,p in model.named_parameters() if p.requires_grad})
    initial_encoder_sha=tree_sha(frozen_backbone_state(model))
    from .model import frozen_decoder_state
    initial_decoder_sha=tree_sha(frozen_decoder_state(model))
    def update(m,o,batch):
        o.zero_grad(set_to_none=True)
        for plan in batch:
            inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(m(**inputs),data,plan,inputs['prompts'])
            assert torch.isfinite(loss);(loss/4).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);o.step()
    # Exact fresh optimizer continuation before proceeding with the budget.
    for step in range(2):update(model,opt,selected[step*4:step*4+4])
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'engineering.pt')
    update(model,opt,selected[8:12]);expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    fresh=build();saved=torch.load(a.output/'engineering.pt',weights_only=True)
    fresh.load_state_dict(saved['model']);fresh.train();fresh_opt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);fresh_opt.load_state_dict(saved['optimizer'])
    update(fresh,fresh_opt,selected[8:12])
    from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
    assert tree_sha(expected)==tree_sha(fresh.state_dict()) and tree_sha(expected_opt)==tree_sha(fresh_opt.state_dict())
    (a.output/'optimizer_acceptance.json').write_text(json.dumps(dict(exact_optimizer_continuation=True,arm=a.arm,seed=a.seed)))
    del fresh,fresh_opt,expected,expected_opt,saved
    for step in range(3,256):
        batch=[selected[(step*4+j)%32] for j in range(4)];update(model,opt,batch)
        if step%16==0:
            status=dict(status='learning',arm=a.arm,updates=step+1,total_updates=256,worker_pid=os.getpid(),seconds=time.monotonic()-started)
            (a.output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    assert a.arm!='aux_detail' or torch.count_nonzero(model.detail_projection.weight).item()>0, 'detail projection did not update'
    assert tree_sha(frozen_decoder_state(model))==initial_decoder_sha, 'frozen-decoder sentinel changed'
    (a.output/'decoder_freeze_acceptance.json').write_text(json.dumps(dict(applicable=False,reason="FewClick heads trainable; no frozen decoder")))
    final_trainable_sha=tree_sha({n:p.detach() for n,p in model.named_parameters() if p.requires_grad})
    assert final_trainable_sha!=initial_trainable_sha, 'no actual trainable parameter change'
    assert tree_sha(frozen_backbone_state(model))==initial_encoder_sha, 'frozen encoder changed'
    (a.output/'parameter_acceptance.json').write_text(json.dumps(dict(trainable_changed=True,frozen_phikon_encoder_unchanged=True,initial_trainable_sha=initial_trainable_sha,final_trainable_sha=final_trainable_sha,encoder_sha=initial_encoder_sha)))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'checkpoint.pt')
    model.eval();rows=[];reference=[]
    with torch.no_grad():
        for plan in selected:
            inputs=data.model_inputs(plan,'xcell','cuda:0');prediction=model(**inputs);prob=prediction['query']['pixel_probability'];gt,mpp=data.labels(plan['query_id']);valid=torch.from_numpy(gt!=255).cuda()
            support_gt,support_mpp=data.labels(plan['support_id']);support_prob=prediction['support']['pixel_probability'][0]
            se=model.encode(model.base.transform(inputs['support']));qe=se if inputs['shared'] else model.encode(model.base.transform(inputs['query']))
            task=model.support_task(se,inputs['prompts']);region=model.base.core.decode(qe,task,inputs['prompts'],inputs['local'])[0]['pixel_probability'][0]
            mechanism=dict(support_target_fraction=float(((support_gt==plan['target'])&(support_gt!=255)).sum()/(support_gt!=255).sum()),support_metrics=score((support_prob>=.5).cpu().numpy(),support_gt==plan['target'],support_gt!=255,support_mpp),region_only_query_metrics=score((region>=.5).cpu().numpy(),gt==plan['target'],gt!=255,mpp),region_probability_mean=float(region[valid].mean()),final_probability_mean=float(prob[0][valid].mean()))
            prompts=dict(inputs['prompts'])
            for k in ['xy','mask']:prompts['positive_'+k],prompts['negative_'+k]=prompts['negative_'+k],prompts['positive_'+k]
            other=model(**dict(inputs,prompts=prompts))['query']['pixel_probability']
            rows.append(dict(plan=plan,mechanism=mechanism,metrics=score((prob[0]>=.5).cpu().numpy(),gt==plan['target'],gt!=255,mpp),valid_spatial_std=float(prob[0][valid].std()),swap_valid_mean=float((prob-other)[0][valid].abs().mean())))
            reference.append(prob.cpu())
    # Reload a new model from the actual final checkpoint, then replay all32.
    del model;torch.cuda.empty_cache();fresh=build();fresh.load_state_dict(torch.load(a.output/'checkpoint.pt',weights_only=True)['model']);fresh.eval()
    with torch.no_grad():
        for plan,expected in zip(selected,reference):assert torch.equal(fresh(**data.model_inputs(plan,'xcell','cuda:0'))['query']['pixel_probability'].cpu(),expected)
    np.savez_compressed(a.output/'masks.npz',masks=np.stack([np.packbits((x[0]>=.5).numpy()) for x in reference]))
    groups={};passed=True
    for d in ['crc','tiger']:
        for kind in ['local','cross_present','cross_absent']:
            g=[r for r in rows if r['plan']['dataset']==d and r['plan']['kind']==kind]
            if kind=='cross_absent':
                values=dict(FPA=float(np.mean([r['metrics']['false_positive_fraction'] for r in g])));ok=values['FPA']<=BOUNDS['absent_group_FPA_max']
            else:
                values=dict(Dice=float(np.mean([r['metrics']['dice'] for r in g])),spatial_std=float(np.mean([r['valid_spatial_std'] for r in g])),median_swap_mean=float(np.median([r['swap_valid_mean'] for r in g])))
                ok=values['Dice']>=BOUNDS['present_group_Dice_min'] and values['spatial_std']>=BOUNDS['present_group_mean_std_min'] and values['median_swap_mean']>=BOUNDS['present_group_median_swap_mean_min']
            region_values=[v['mechanism']['region_only_query_metrics'] for v in g]
            region_ok=(float(np.mean([v['false_positive_fraction'] for v in region_values]))<=BOUNDS['region_absent_group_FPA_max']) if kind=='cross_absent' else (float(np.mean([v['dice'] for v in region_values]))>=BOUNDS['region_present_group_Dice_min'])
            values['region_mean_Dice']=float(np.mean([v['dice'] for v in region_values]))
            values['region_functional_passed']=region_ok;ok=ok and region_ok
            groups[d+'|'+kind]=dict(n=len(g),values=values,passed=ok);passed=passed and ok
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_repaired_functional_gate' if passed else 'failed_repaired_functional_gate',arm=a.arm,seed=a.seed,groups=groups,rows=rows,bounds=BOUNDS,exact_optimizer_continuation=True,fresh_final_predictions_exact=True,train_only=True,source_sha256=sha(__file__),seconds=time.monotonic()-started,checkpoint_sha256=sha(a.output/'checkpoint.pt')),indent=2))
    if not passed:raise RuntimeError('Repaired learning gate failed; stop without recipe/threshold retries')


if __name__=='__main__':main()
