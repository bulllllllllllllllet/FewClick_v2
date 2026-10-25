"""Inspect saved parameter changes and absence of historical J5 initialization."""
import argparse
import gc
import json
from pathlib import Path

import torch

from .adapted_models import build,COCO,COCO_SHA
from .audit import digest,write
from .local_data import LocalData,fewclick_inputs
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor


def verify(gates,output,device):
    torch.set_num_threads(4);records={};task_hashes=set();public=torch.load(COCO,map_location='cpu',weights_only=False)
    if digest(COCO)!=COCO_SHA:raise ValueError('generic COCO weight changed')
    for directory in gates:
        summary=json.loads((directory/'summary.json').read_text());run=json.loads((directory/'run.json').read_text())
        if (summary['status']!='complete' or not summary['gate'] or not summary['fresh_gate_resume']
                or not summary['checkpoint_reload_exact'] or not summary['nonzero_joint_backbone_gradient']):raise ValueError('gate incomplete')
        task_hashes.add(summary['training_consumed_sha256']);method=summary['method']
        before=torch.load(directory/'checkpoint_step_00002.pth',map_location='cpu',weights_only=False)['model']
        after=torch.load(directory/'checkpoint_step_00004.pth',map_location='cpu',weights_only=False)['model']
        changed=[name for name in before if not torch.equal(before[name],after[name])]
        backbone=[name for name in changed if '.backbone.' in name or '.image_encoder.' in name]
        heads=[name for name in changed if name not in backbone]
        if not backbone or not heads:raise ValueError('actual backbone/head parameters did not change')
        record=dict(actual_changed_tensors=len(changed),backbone_changed_tensors=len(backbone),head_changed_tensors=len(heads),
            step2_checkpoint_sha256=digest(directory/'checkpoint_step_00002.pth'),step4_checkpoint_sha256=digest(directory/'checkpoint_step_00004.pth'))
        if method.startswith('fewclick_'):
            checked=0
            for scale in range(3):
                prefix=f'core.encoders.{scale}.'
                for name,value in public.items():
                    if name.startswith('backbone.'):
                        if prefix+name not in before or not torch.equal(value,before[prefix+name]):raise ValueError('non-generic warmup backbone')
                        checked+=1
                if not any(name.startswith(prefix+'backbone.') for name in backbone):raise ValueError('scale received no verified backbone update')
            if run['contract']['initialization']['J5_loaded']:raise ValueError('historical J5 loaded')
            record.update(generic_backbone_tensors_checked=checked,all_three_backbones_changed=True,J5_loaded=False)
        if method=='fewclick_m3d_image':
            data=LocalData(Path(run['args']['tasks']),Path(run['args']['realized']));plan=data.training[0]
            model=build(method,device);model.load_state_dict(after,strict=True);model.eval()
            raw,prompts,_=data.materialize(plan,device,True,True)
            # Check GPU normalization against the established image transform.
            reference=he_to_tensor(raw['rgb'][0].permute(1,2,0).cpu().numpy())[None].to(device)
            if not torch.equal(fewclick_inputs(raw)['image'],reference):raise ValueError('FewClick image normalization changed')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                expected=model(raw,prompts)['pixel_probability']
                altered={k:torch.randn_like(v) if k.startswith(('cells','total')) else ~v if k.startswith('cell_valid') else v for k,v in raw.items()}
                actual=model(altered,prompts)['pixel_probability']
            if not torch.equal(expected,actual):raise ValueError('image-only control depends on cell evidence')
            record.update(cell_perturbation_output_identical=True,normalization_identical=True)
            del model,raw,prompts,expected,actual;gc.collect();torch.cuda.empty_cache()
        records[method]=record;del before,after;gc.collect()
    if len(task_hashes)!=1:raise ValueError('gate methods consumed different actual clicks')
    result=dict(status='passed',methods=records,shared_training_coordinate_sha256=next(iter(task_hashes)),
        limits='4-update gates establish executable gradients/resume only; not convergence or performance',generic_coco_sha256=COCO_SHA)
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gates',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.gates,a.output,torch.device('cuda:0'))
