"""Preregistered train-only baseline learning check; never a validation selector."""
import argparse,json,os
from pathlib import Path
import pyvips
import numpy as np
import torch
from .data import BridgeData,BASE
from .vocabulary_model import VocabularyFewClick
from .restored_loss import episode_loss
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['continuous','fixed','ema'],required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=BridgeData();source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';plans=list(map(json.loads,source.read_text().splitlines()));selected=[]
    for d in ['crc','tiger']:selected.extend([p for p in plans if p['dataset']==d and p['kind']=='local'][:16])
    assert len(selected)==32
    model=VocabularyFewClick(a.mode);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    for step in range(256):
        opt.zero_grad(set_to_none=True);loss_sum=0.
        for j in range(4):
            plan=selected[(step*4+j)%32];inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(model(**inputs),data,plan,inputs['prompts']);(loss/4).backward();loss_sum+=float(loss.detach())/4
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);opt.step();model.after_optimizer_step()
        if step%32==0:print(json.dumps(dict(updates=step+1,loss=loss_sum)),flush=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),a.output/'checkpoint.pt')
    model.eval();rows=[]
    with torch.no_grad():
        for plan in selected:
            inputs=data.model_inputs(plan,'xcell','cuda:0');out=model(**inputs);prob=out['query']['pixel_probability'];gt,mpp=data.labels(plan['query_id']);mask=(prob[0]>=.5).cpu().numpy()
            prompts=dict(inputs['prompts'])
            for suffix in ['xy','mask']:prompts['positive_'+suffix],prompts['negative_'+suffix]=prompts['negative_'+suffix],prompts['positive_'+suffix]
            swapped=model(**dict(inputs,prompts=prompts))['query']['pixel_probability']
            rows.append(dict(plan=plan,metrics=score(mask,gt==plan['target'],gt!=255,mpp),spatial_std=float(prob.std()),swap_change=float((swapped-prob).abs().max())))
    mean_dice=float(np.mean([r['metrics']['dice'] for r in rows]));std=float(np.mean([r['spatial_std'] for r in rows]));swap=max(r['swap_change'] for r in rows)
    passed=mean_dice>=.2 and std>=.02 and swap>=.05
    result=dict(status='accepted_train_only_functional_gate' if passed else 'failed_train_only_functional_gate',worker_pid=os.getpid(),mode=a.mode,train_mean_Dice=mean_dice,mean_spatial_std=std,max_prompt_swap_change=swap,updates=256,train_only=True,rows=rows,bounds=dict(train_Dice_min=.2,spatial_std_min=.02,swap_min=.05),source_sha256=sha(__file__),not_used_as_formal_initialization=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if not passed:raise RuntimeError('Baseline functional learning failed; stop before full four-arm retraining')


if __name__=='__main__':main()
