"""Fixed-checkpoint bridge collapse diagnosis; no optimization or threshold selection."""
import argparse,json,os
from pathlib import Path
import pyvips
import numpy as np
import torch
from .data import BridgeData,BASE
from .deterministic_model import build_bridge_image
from .gate_four_deterministic import ARMS
from benchmarks.v5.encoder_research.train_cell_only_pilot import gate_indices
from benchmarks.v5.encoder_research.continuous_data import ContinuousData,sha


def stats(value):
    x=value.detach().float()
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=BridgeData();source=BASE/'phikon_joint_tasks_1031_222610/data/training4000.jsonl';lines=source.read_text().splitlines();training=[json.loads(lines[i]) for i in gate_indices(lines)]
    selected=[('train','recorded',p) for p in training];seen=set()
    for scope,budget,plan in ContinuousData.validation(data):
        if budget!='3+2':continue
        key=(scope,plan['dataset'],plan['kind'].endswith('absent'))
        if key not in seen:seen.add(key);selected.append((scope,budget,plan))
    results={}
    for arm,(image_arm,cell_arm) in ARMS.items():
        model,identity=build_bridge_image(image_arm);checkpoint=BASE/'bridge_training_1101_155846'/arm/'checkpoint.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True);del state
        traces={};handles=[]
        def hook(name):
            def capture(module,args,out):
                if isinstance(out,torch.Tensor):
                    traces[name]=stats(out)
                    if out.ndim==3:traces[name]['between_node_std']=float(out.detach().float().std(dim=1).mean())
            return capture
        for i,e in enumerate(model.encoders):handles.append(e.embedding.register_forward_hook(hook('embedding_'+str(i))))
        for i,b in enumerate(model.decoder.blocks):handles.append(b.register_forward_hook(hook('graph_'+str(i))))
        handles.append(model.decoder.delta_head.register_forward_hook(hook('node_logits')))
        rows=[]
        with torch.no_grad():
            for scope,budget,plan in selected:
                inputs=data.model_inputs(plan,cell_arm,'cuda:0');traces.clear();output=model(**inputs);prob=output['query']['pixel_probability'];original_traces=dict(traces)
                gt,_=data.labels(plan['query_id']);valid=torch.from_numpy(gt!=255)[None].cuda();positive=torch.from_numpy(gt==plan['target'])[None].cuda()
                row=dict(scope=scope,dataset=plan['dataset'],kind=plan['kind'],query_id=plan['query_id'],support_id=plan['support_id'],target=plan['target'],probability=stats(prob),predicted_fraction=float((prob[valid]>=.5).float().mean()),GT_fraction=float(positive[valid].float().mean()),activation=original_traces,
                         positive_probability_mean=float(prob[valid&positive].mean()) if (valid&positive).any() else None,
                         negative_probability_mean=float(prob[valid&~positive].mean()) if (valid&~positive).any() else None)
                se=model.encode_image(inputs['support']);assign=se['assignment_low'].float();tokens=se['tokens'][0].float()
                row['assignment']=dict(entropy_mean=float((-(assign*assign.clamp_min(1e-12).log()).sum(1)).mean()),area_min=float(assign.mean((2,3)).min()),area_max=float(assign.mean((2,3)).max()),token_between_region_std=float(tokens.std(1).mean()))
                swapped=dict(inputs);prompts=dict(inputs['prompts'])
                for suffix in ['xy','mask']:
                    prompts['positive_'+suffix],prompts['negative_'+suffix]=prompts['negative_'+suffix],prompts['positive_'+suffix]
                swapped['prompts']=prompts;counter=model(**swapped)
                row['swapped_prompts']=dict(probability_max_change=float((counter['query']['pixel_probability']-prob).abs().max()),task_token_max_change=float((counter['task_token']-output['task_token']).abs().max()))
                if arm=='A' and scope=='train':
                    blocks=model.decoder.blocks;model.decoder.blocks=torch.nn.ModuleList([])
                    bypass=model(**inputs)['query']['pixel_probability'];model.decoder.blocks=blocks
                    row['bypass_graph']=dict(probability=stats(bypass),predicted_fraction=float((bypass[valid]>=.5).float().mean()))
                    altered=dict(inputs)
                    for side in ['support','query']:
                        altered[side]={k:v.clone() for k,v in inputs[side].items()}
                        for scale in range(3):
                            key='cells' if scale==0 else 'cells_'+str(scale);cells=altered[side][key]
                            cells[:,:,10:]=cells[:,:,10:]*cells.new_tensor(data.std)+cells.new_tensor(data.mean)
                    raw=model(**altered)['query']['pixel_probability'];row['raw_XCell_counterfactual']=dict(probability=stats(raw),max_change=float((raw-prob).abs().max()))
                rows.append(row)
                print(json.dumps(dict(arm=arm,completed=len(rows),total=len(selected))),flush=True)
        results[arm]=dict(checkpoint_sha256=sha(checkpoint),rows=rows)
        for h in handles:h.remove()
        (a.output/(arm+'.json')).write_text(json.dumps(results[arm],indent=2)+'\n')
        del model;torch.cuda.empty_cache()
    (a.output/'report.json').write_text(json.dumps(dict(status='complete_fixed_checkpoint_collapse_diagnosis',worker_pid=os.getpid(),arms=results,source_sha256=sha(__file__),training_updates=0,threshold_changed=False,scope='fixed representative task observations and inference interventions; not causal repair validation'),indent=2)+'\n')


if __name__=='__main__':main()
