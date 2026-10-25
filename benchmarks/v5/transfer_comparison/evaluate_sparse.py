"""Complete sparse-support transfer evaluation with frozen feature caches."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
import yaml

from .data import TransferData
from .prototypes import dense_prototype,region_prototype
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import fewclick_inputs
from benchmarks.v5.fair_comparison.metrics import score,VERSION
from benchmarks.v5.fair_comparison.protocol import nested


def summarize(rows):
    result={}
    for budget in sorted({r['click_budget'] for r in rows}):
        selected=[r for r in rows if r['click_budget']==budget]
        strata=defaultdict(list)
        for r in selected:
            strata['overall'].append(r);strata[r['dataset']].append(r);strata[r['protocol']].append(r)
            strata[r['dataset']+'|'+r['kind']].append(r)
        result[budget]={}
        for label,values in sorted(strata.items()):
            present=[r for r in values if r['positive']>0];absent=[r for r in values if r['positive']==0]
            bf=[r['bf1_um2'] for r in present if r['bf1_um2'] is not None]
            pairs=defaultdict(list)
            for r in present:pairs[r['pair_id']].append(r['dice'])
            mean=lambda x:float(np.mean(x)) if x else None
            result[budget][label]=dict(tasks=len(values),present=len(present),absent=len(absent),
                present_dice=mean([np.mean(v) for v in pairs.values()]),
                precision=mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0. for r in present]),
                recall=mean([r['tp']/(r['tp']+r['fn']) for r in present]),bf1_um2=mean(bf),bf1_denominator=len(bf),
                absent_FPA=mean([r['false_positive_fraction'] for r in absent]),
                absent_FPA_ge_1percent=mean([r['false_positive_fraction']>=.01 for r in absent]),
                absent_FPA_ge_5percent=mean([r['false_positive_fraction']>=.05 for r in absent]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--method',choices=['retained_m3','sam_public_features'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--gate',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(4);device=torch.device(a.device)
    try:
        data=TransferData(a.tasks)
        plans=data.plans
        if a.gate:
            plans=[next(p for p in data.plans if p['dataset']==d and p['query_presence']==presence and p['prompt_group']==0)
                   for d in sorted({p['dataset'] for p in data.plans}) for presence in ('present','absent')]
        if a.method=='retained_m3':
            from benchmarks.v5.paper_multiscale.model import load_model
            if a.checkpoint is None:raise ValueError('explicit frozen M3 checkpoint required')
            model,prior=load_model(device,True,False);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
            if saved['contract']['arm']!='multi_region' or saved['contract']['catalog_sha256']!=data.contract['catalog_sha256']:raise ValueError('retained model/catalog mismatch')
            model.load_state_dict(saved['model'],strict=True);del saved;model.eval()
            methods=['retained_m3_graph','retained_m3_same_feature_prototype']
            prior.update(checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),
                scope='matched-feature diagnostic at retained J5 weights; historical supervision; not retrained architecture ablation')
            @lru_cache(maxsize=64)
            def encode(identity):
                raw={k:v.to(device) for k,v in data.inputs(identity,True,True).items()}
                return model.encode_image(fewclick_inputs(raw))
        else:
            from benchmarks.v4.baseline.adapters.sam_adapter import SAMAdapter
            from segment_anything.utils.transforms import ResizeLongestSide
            config=yaml.safe_load(Path('benchmarks/v4/baseline/configs/sam_zero_shot.yaml').read_text());config['device']=str(device)
            adapter=SAMAdapter(config);model=adapter.model;transform=ResizeLongestSide(model.image_encoder.img_size)
            prior=adapter.provenance()|dict(scope='custom frozen SAM feature retrieval, not native SAM remote inference')
            methods=['sam_public_feature_prototype']
            @lru_cache(maxsize=64)
            def encode(identity):
                rgb=data.inputs(identity,False,False)['rgb'][0].permute(1,2,0).numpy()
                image=transform.apply_image(rgb);tensor=torch.as_tensor(image,device=device).permute(2,0,1)
                return model.image_encoder(model.preprocess(tensor)[None])
        folders={name:a.output/(name+'_'+a.output.name.rsplit('_',2)[-2]+'_'+a.output.name.rsplit('_',2)[-1]) for name in methods}
        handles={};rows={name:[] for name in methods}
        provenance=dict(worker_pid=os.getpid(),device=str(device),gate=a.gate,protocol_sha256=digest(a.tasks/'contract.json'),
            tasks=str(a.tasks),prior=prior,metric_version=VERSION,threshold=.5,prototype_temperature=.1,prototype_margin=0.,
            query_human_clicks=0,query_class_ID_in_model=False,query_GT_in_model=False,cache_size=64,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            sources={str(f):digest(f) for f in Path(__file__).parent.glob('*.py')},
            cost_scope='shared encoder feature cache; per-query method latency excludes encoding, IO and scoring; total job wall time also recorded')
        for name,folder in folders.items():
            folder.mkdir();write(folder/'run.json',provenance|dict(method=name));handles[name]=(folder/'rows.jsonl').open('x')
        write(a.output/'run.json',provenance|dict(methods=methods))
        with torch.inference_mode():
            for maximum in plans:
                # Only RGB/cell inputs and support point tensors reach inference.
                source=encode(maximum['support_id']);query=encode(maximum['query_id'])
                for budget in data.contract['report_budgets']:
                    plan=nested(maximum,budget);prompts={}
                    for sign in ('positive','negative'):
                        prompts[sign+'_xy']=torch.tensor([plan[sign+'_xy']],device=device,dtype=torch.float32)
                        prompts[sign+'_mask']=torch.ones(1,len(plan[sign+'_xy']),device=device,dtype=torch.bool)
                    prompts['prompt_size_id']=torch.ones(1,device=device,dtype=torch.long)
                    tick=time.perf_counter()
                    if a.method=='retained_m3':
                        task=model.support_task(source,prompts);native,maps=model.decode(query,task,prompts,False)
                        if maps.count_nonzero():raise ValueError('remote query received source-coordinate maps')
                        prototype=region_prototype(source,query,prompts);outputs=dict(zip(methods,[native,prototype]))
                    else:outputs={methods[0]:dense_prototype(source,query,prompts)}
                    if device.type=='cuda':torch.cuda.synchronize()
                    elapsed=1000*(time.perf_counter()-tick)
                    # GT is first requested after every prediction has been made.
                    supervision=data.supervision(plan)
                    for name,out in outputs.items():
                        probability=out['pixel_probability'][0].float().cpu().numpy()
                        if probability.shape!=(512,512) or not np.isfinite(probability).all():raise ValueError('invalid sparse transfer probability')
                        metrics=score(probability>=.5,supervision['truth'],supervision['valid'],supervision['mpp'])
                        meta=data.rows[plan['query_id']]
                        row=plan|metrics|dict(method=name,latency_ms=elapsed,latency_scope='joint decode+retrieval' if len(outputs)>1 else 'retrieval',
                            space_key=meta['space_key'],source_id=meta['source_id'],spatial_group_id=meta['spatial_group_id'])
                        rows[name].append(row);handles[name].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                if len(rows[methods[0]])%100==0:
                    for handle in handles.values():handle.flush()
                    state=dict(status='running',worker_pid=os.getpid(),rows_per_method=len(rows[methods[0]]),total_rows=4*len(plans),
                        cache=encode.cache_info()._asdict(),seconds=time.time()-started)
                    write(a.output/'status.json',state);print(json.dumps(state),flush=True)
        for handle in handles.values():handle.close()
        summaries={}
        for name,folder in folders.items():
            result=dict(status='complete',method=name,gate=a.gate,rows=len(rows[name]),metrics=summarize(rows[name]),rows_sha256=digest(folder/'rows.jsonl'),
                protocol_sha256=digest(a.tasks/'contract.json'),scope=prior['scope'])
            write(folder/'summary.json',result);summaries[name]=str(folder)
        result=dict(status='complete',worker_pid=os.getpid(),gate=a.gate,outputs=summaries,rows_per_method=4*len(plans),
            cache=encode.cache_info()._asdict(),seconds=time.time()-started)
        write(a.output/'summary.json',result);write(a.output/'status.json',result);print(json.dumps(result),flush=True)
    except Exception:
        write(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
