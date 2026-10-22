"""Real dynamic episodes, frozen replay, gradient/update and reload acceptance."""
import argparse
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
import torch

from benchmarks.v5.multidomain_data.prepare import dump,sha
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import encode_image,encode_support,query_features
from .episodes import EpisodeCorpus,DATASETS,KINDS
from .evaluate import validation_plans
from .losses import episode_loss,mask_loss
from .model import load_model
from .train import grad_norm


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0');started=time.time();torch.manual_seed(20261021)
    try:
        corpus=EpisodeCorpus(a.catalog);dump(a.output/'coverage.json',corpus.coverage)
        plans=validation_plans(corpus);dump(a.output/'validation_plans.json',plans)
        model,weights=load_model(device,mapper='centroid');first=corpus.plan(90,dataset='crc',kind='local',quality_aug=False)
        inputs,supervision=corpus.materialize(first,device)
        with torch.inference_mode():
            old=encode_image(model.j5,inputs['support']);new=model.encode_image(inputs['support'])
            feature_error=max(float((old[k]-new[k]).abs().max()) for k in old if old[k].dtype!=torch.bool)
            task=encode_support(model.j5,old,inputs['prompts']);f=query_features(model.j5,old,task,inputs['prompts'],query_id='same',support_id='same')
            ref=model.dense(f)['pixel_probability'];actual=model(**inputs,with_support=False)['query']['pixel_probability']
            replay_error=float((ref-actual).abs().max())
        if max(feature_error,replay_error)>2e-6:raise ValueError('retained frozen bridge replay mismatch')
        model.mapper='pixel';model.set_stage(True);optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
        modules=dict(prompt=model.j5.decoder.prompt_model,cell=model.j5.cell,dense=model.dense,assignment=model.j5.phase2.assignment,layer3=model.j5.phase2.backbone.layer3,layer4=model.j5.phase2.backbone.layer4)
        probes={name:next(m.parameters()) for name,m in modules.items()};before={name:p.detach().clone() for name,p in probes.items()}
        bn_before={n:b.clone() for n,b in model.named_buffers() if 'running_' in n};rows=[];max_grad={name:0. for name in modules}
        train_datasets=corpus.training_policy['datasets'] if corpus.training_policy else DATASETS
        for d,dataset in enumerate(train_datasets):
            for k,kind in enumerate(KINDS):
                plan=corpus.plan(1000+d*10+k,dataset=dataset,kind=kind)
                inputs,supervision=corpus.materialize(plan,device);model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(**inputs);loss,parts=episode_loss(output,supervision,inputs['prompts'],inputs['local'])
                if not inputs['local'] and output['query_prompt_maps'].count_nonzero():raise ValueError('source heatmap leaked to remote query')
                loss.backward();norms={name:grad_norm(module) for name,module in modules.items()}
                if not all(np.isfinite(v) for v in norms.values()):raise FloatingPointError('non-finite real gradient')
                for name,value in norms.items():max_grad[name]=max(max_grad[name],value)
                total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(total_norm):raise FloatingPointError('global gradient')
                optimizer.step();row=dict(plan,loss=float(loss.detach()),gradient_norms=norms,positive_query_pixels=int((supervision['truth']&supervision['valid']).sum()))
                rows.append(row);dump(a.output/f'case_{len(rows):02d}.json',row);print('REAL_SMOKE',dataset,kind,row['loss'],flush=True)
        if any(v<=0 for v in max_grad.values()):raise ValueError('required trainable path has zero gradient')
        changes={name:float((p-before[name]).abs().max()) for name,p in probes.items()}
        if any(v<=0 for v in changes.values()):raise ValueError('required module parameters did not change')
        if any(not torch.equal(dict(model.named_buffers())[name],value) for name,value in bn_before.items()):raise ValueError('frozen BN statistics changed')
        model.eval();plan=corpus.plan(5000,dataset=train_datasets[-1],kind='cross_present');inputs,supervision=corpus.materialize(plan,device)
        with torch.inference_mode():
            clean=model(**inputs,with_support=False)['query']['pixel_probability']
            contaminated={**inputs,'query':inputs['query']|{'pixel_gt':torch.zeros_like(supervision['truth']),'target_class':torch.tensor([999],device=device)},'prompts':inputs['prompts']|{'target_class':torch.tensor([999],device=device)}}
            repeated=model(**contaminated,with_support=False)['query']['pixel_probability']
            if not torch.equal(clean,repeated):raise ValueError('query GT/target leaked into model forward')
            swapped=dict(inputs['prompts'])
            for suffix in ['xy','mask']:swapped['positive_'+suffix],swapped['negative_'+suffix]=swapped['negative_'+suffix],swapped['positive_'+suffix]
            altered=model(**(inputs|dict(prompts=swapped)),with_support=False)['query']['pixel_probability']
            prompt_difference=float((altered-clean).abs().max())
            if prompt_difference<=1e-6:raise ValueError('model ignores changed source prompt')
            empty={k:(v[:,:0] if k in ['cells','cell_valid'] else torch.zeros_like(v) if k=='total_cell_count' else v) for k,v in inputs['support'].items()}
            empty_out=model(**(inputs|dict(support=empty)),with_support=False)['query']['pixel_probability']
            if not torch.isfinite(empty_out).all():raise ValueError('empty cell inputs fail')
        # Full-target is a mathematical edge-case test, never a changed real GT score.
        logits=torch.zeros_like(supervision['truth'],dtype=torch.float32,requires_grad=True)
        full,_=mask_loss(dict(pixel_logits=logits),supervision['valid'],supervision['valid'],supervision['mpp']);full.backward()
        if not torch.isfinite(full) or not logits.grad.abs().sum()>0:raise ValueError('full-target supervision has no signal')
        path=a.output/'checkpoint.pth';torch.save(dict(model=model.state_dict(),mapper='pixel'),path)
        reloaded,_=load_model(device,mapper='pixel');reloaded.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model'],strict=True);reloaded.eval()
        with torch.inference_mode():prediction=reloaded(**inputs,with_support=False)['query']['pixel_probability']
        if not torch.equal(clean,prediction):raise ValueError('fresh checkpoint reload changed prediction')
        report=dict(status='passed',worker_pid=os.getpid(),seconds=time.time()-started,real_episodes=rows,max_gradient_norms=max_grad,max_parameter_changes=changes,
            weights=weights,frozen_bridge_feature_error=feature_error,frozen_bridge_prediction_error=replay_error,prompt_max_prediction_change=prompt_difference,
            query_GT_independent=True,remote_query_heatmaps_zero=True,BN_statistics_frozen=True,empty_cells_finite=True,full_target_loss_finite=True,
            fresh_checkpoint_reload_exact=True,puma_used=False,test_used=False,
            training_policy=corpus.training_policy,training_policy_sha256=corpus.training_policy_sha256,
            source_sha256={str(f):sha(f.read_bytes()) for f in sorted(Path(__file__).parent.glob('*.py'))})
        dump(a.output/'summary.json',report);dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('DYNAMIC_SMOKE_PASSED',flush=True)
    except Exception as exc:dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
