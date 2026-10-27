"""Probe actual selected-model outputs and actual seen tasks for excluded-label loss."""
import argparse
import hashlib
import json
from pathlib import Path
import os

import numpy as np
from PIL import Image
import torch

from .audit_heldout_local import read, sha, require
from .diagnose_adipose import load_model
from benchmarks.v5.heldout_comparison.training import TrainingData
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.train_adapted import loss_for


def run(root, output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0')
    train=root/'formal_queue_1026_012700/train_adipose_fewclick_m3d_image_1026_012700'
    meta=read(train/'run.json');tasks=Path(meta['args']['tasks']);realized=Path(meta['args']['realized'])
    require(sha(str(realized/'training_plans.jsonl'))==meta['contract']['training_plans_sha256'],'actual training SHA')
    data=TrainingData(tasks,realized);original=LocalData(root/'protocols_geometry_1026_010300/adipose/heldout_local')
    chosen={}
    for plan in data.training:
        domain=plan['dataset'];excluded=10 if domain=='crc' else 7
        if domain not in chosen and json.loads(original.rows[plan['query_id']]['counts_json']).get(str(excluded),0)>100:
            chosen[domain]=plan
        if len(chosen)==2:break
    require(set(chosen)=={'crc','tiger'},'representative actual domains')
    best=read(train/'best_checkpoint.json');require(sha(best['path'])==best['checkpoint_sha256'],'selected weight SHA')
    model=load_model(Path(best['path']),'fewclick_m3d_image',device)
    checks=[]
    for domain,plan in chosen.items():
        excluded=10 if domain=='crc' else 7
        raw,prompts,supervision=data.materialize(plan,device,model.contexts,model.cells)
        original_labels=np.asarray(Image.open(original.rows[plan['query_id']]['labels']))
        old_mask=np.isin(original_labels,[excluded]);masked=supervision['labels'][0].cpu().numpy()
        expected=original_labels.copy();expected[old_mask]=255
        require(np.array_equal(masked,expected),'actual masked GT is only original excluded label->255')
        require(not bool(supervision['valid'][0].cpu().numpy()[old_mask].any()),'excluded pixels have valid false')
        point_values={}
        for sign in ('positive','negative'):
            xy=np.asarray(plan[sign+'_xy']);pixel=np.floor(xy*np.array(masked.shape[::-1])).astype(int)
            values=original_labels[pixel[:,1],pixel[:,0]]
            require(not np.isin(values,[excluded,255]).any(),'actual points avoid excluded labels')
            point_values[sign]=values.tolist()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):result=model(raw,prompts)
        logits=result['pixel_logits'].detach().float().clone().requires_grad_()
        isolated={k:v.detach() if torch.is_tensor(v) else v for k,v in result.items()}
        isolated['pixel_logits']=logits;isolated['pixel_probability']=logits.sigmoid()
        loss=loss_for(model,isolated,supervision,prompts)
        gradient,=torch.autograd.grad(loss,logits)
        invalid=torch.as_tensor(old_mask,device=device)[None]
        altered=dict(supervision);altered['truth']=torch.where(invalid,~supervision['truth'],supervision['truth'])
        changed=loss_for(model,isolated,altered,prompts)
        require(torch.equal(loss,changed),'excluded truth cannot change complete output loss')
        require(int(torch.count_nonzero(gradient[invalid]))==0,'excluded output-logit gradients zero')
        checks.append(dict(dataset=domain,plan=plan,excluded_label=excluded,excluded_pixels=int(old_mask.sum()),
            excluded_valid_pixels=0,excluded_nonzero_output_gradients=0,point_original_labels=point_values,
            loss=float(loss.detach()),loss_after_excluded_truth_flip=float(changed.detach()),parameter_updates=0))
    report=dict(status='passed_actual_two_domain_selected_model_loss_probes',worker_pid=os.getpid(),checks=checks,
        checkpoint_sha256=best['checkpoint_sha256'],training_coordinate_sha256=meta['contract']['training_plans_sha256'],
        source_sha256=sha(__file__),
        limits=['Two representative actual tasks; not all40k gradient replay','Output-logit loss gradients only; label-free assignment regularization can still learn from RGB morphology'])
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=[{k:r[k] for k in ['dataset','excluded_pixels','excluded_nonzero_output_gradients','loss']} for r in checks])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
