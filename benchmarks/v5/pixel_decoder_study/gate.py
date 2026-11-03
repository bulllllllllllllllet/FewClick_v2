"""Actual decoder invariants, optimizer replay and per-arm GPU throughput."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import torch

from .model import DecoderData, DecoderModel, ARMS
from benchmarks.v5.architecture_bridge.spatial_controls import SpatialModel
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    settings=json.loads((root/'settings.json').read_text());out=root/'engineering';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
    plans=fixed_train_plans(source)
    selected=[next(p for p in plans if p['dataset']==d and p['kind']==k)
              for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']]
    data=DecoderData(settings['cache'],'region');reports={};common=None
    def progress(**kw):
        kw['worker_pid']=os.getpid();(out/'status.json').write_text(json.dumps(kw));print(json.dumps(kw),flush=True)
    def update(model,opt,batch):
        opt.zero_grad(set_to_none=True)
        for plan in batch:
            inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(model(**inputs),data,plan,inputs['prompts'])
            assert torch.isfinite(loss);(loss/len(batch)).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
    legacy=SpatialModel('I').eval();expected_common=tree_sha(legacy.state_dict());references=[]
    with torch.no_grad():
        for plan in selected:
            inputs=data.model_inputs(plan,'xcell','cuda:0');o=legacy(**inputs)
            references.append(dict(logits=o['query']['pixel_logits'].cpu(),support=o['support']['pixel_logits'].cpu(),
                                   probability=o['query']['pixel_probability'].cpu(),loss=float(episode_loss(o,data,plan,inputs['prompts']))))
    del legacy;gc.collect();torch.cuda.empty_cache()
    for arm in ARMS:
        model=DecoderModel(arm);model.eval();state=tree_sha(model.base.state_dict());assert state==expected_common
        if common is None:common=state
        assert state==common
        probability_error=0.
        with torch.no_grad():
            for plan,reference in zip(selected,references):
                inputs=data.model_inputs(plan,'xcell','cuda:0');output=model(**inputs)
                assert torch.equal(output['query']['pixel_logits'].cpu(),reference['logits'])
                assert torch.equal(output['support']['pixel_logits'].cpu(),reference['support'])
                error=float((output['query']['pixel_probability'].cpu()-reference['probability']).abs().max());probability_error=max(error,probability_error);assert error<=2e-7
                assert abs(float(episode_loss(output,data,plan,inputs['prompts']))-reference['loss'])<=2e-6
                if plan['support_id']!=plan['query_id']:assert not output['query_prompt_maps'].any()
                changed=dict(inputs)
                for name in ['support','query']:
                    changed[name]={k:(v+17 if k.startswith('cells') or k.startswith('total') else v) for k,v in inputs[name].items()}
                assert torch.equal(model(**changed)['query']['pixel_logits'],output['query']['pixel_logits'])
        if arm=='region':
            # The wrapper must preserve the original gradient path and update exactly.
            original=SpatialModel('I');o=torch.optim.AdamW([p for p in original.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
            twin=DecoderModel('region');t=torch.optim.AdamW([p for p in twin.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
            original.train();twin.train()
            for batch in [plans[:4],plans[4:8]]:update(original,o,batch);update(twin,t,batch)
            assert tree_sha(original.state_dict())==tree_sha(twin.base.state_dict())
            assert tree_sha(o.state_dict())==tree_sha(t.state_dict())
            del original,twin,o,t;gc.collect();torch.cuda.empty_cache()
        initial=tree_sha(model.base.state_dict());model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        update(model,opt,plans[:4]);update(model,opt,plans[4:8])
        path=out/(arm+'_continuation.pt');torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),path)
        update(model,opt,plans[8:12]);expected=tree_sha(model.state_dict());expected_opt=tree_sha(opt.state_dict())
        fresh=DecoderModel(arm);saved=torch.load(path,map_location='cpu',weights_only=True);fresh.load_state_dict(saved['model']);fresh.train()
        fopt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);fopt.load_state_dict(saved['optimizer'])
        update(fresh,fopt,plans[8:12]);assert tree_sha(fresh.state_dict())==expected and tree_sha(fopt.state_dict())==expected_opt
        if arm=='pixel':
            assert model.pixel.output.weight.grad is not None and model.pixel.output.weight.grad.abs().sum()>0
            assert model.pixel.rgb2[0].weight.grad is not None and model.pixel.rgb2[0].weight.grad.abs().sum()>0
            model.eval();inputs=data.model_inputs(selected[1],'xcell','cuda:0');model.pixel_enabled=False
            with torch.no_grad():assert torch.equal(model(**inputs)['query']['pixel_logits'],model.base(**inputs)['query']['pixel_logits'])
            model.pixel_enabled=True;model.train()
        del fresh,fopt,saved;gc.collect();torch.cuda.empty_cache()
        timings=[]
        for step in range(20):
            torch.cuda.synchronize();start=time.monotonic();batch=[plans[(step*4+j)%32] for j in range(4)]
            update(model,opt,batch);torch.cuda.synchronize();timings.append(time.monotonic()-start)
            progress(stage='throughput',arm=arm,updates=step+1,total=20)
        model.eval();infer=[]
        with torch.no_grad():
            for plan in (selected*17)[:100]:
                inputs=data.model_inputs(plan,'xcell','cuda:0');torch.cuda.synchronize();start=time.monotonic()
                model(**inputs);torch.cuda.synchronize();infer.append(time.monotonic()-start)
        report=dict(status='accepted_pixel_decoder_GPU_probe',arm=arm,updates=20,inference_tasks=len(infer),train_seconds=timings,
                    inference_seconds=infer,initial_common_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),
                    registered_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    active_optimizer_parameters=sum(p.numel() for g in opt.param_groups for p in g['params'] if p in opt.state),
                    pixel_parameters=sum(p.numel() for p in model.pixel.parameters()) if model.pixel is not None else 0,
                    initial_logits_exact=True,initial_probability_max_error=probability_error,exact_optimizer_continuation=True,
                    legacy_wrapper_updates_exact=arm=='region',cell_inputs_irrelevant=True,no_remote_query_prompts=True,
                    source_sha256=sha(__file__),worker_pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        reports[arm]=report;(out/(arm+'_benchmark.json')).write_text(json.dumps(report,indent=2)+'\n')
        del model,opt;gc.collect();torch.cuda.empty_cache()
    result=dict(status='accepted_pixel_decoder_engineering',common_initial_sha256=common,selected_tasks=selected,arms=reports,
                train_source_sha256=sha(source),source_sha256=sha(__file__),learning_accepted=False)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');progress(stage='complete')


if __name__=='__main__':main()
