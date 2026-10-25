"""Local-adapted transfer without target-data initialization or query supervision."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import pyvips
import numpy as np
import torch

from .data import TransferData
from .prototypes import dense_prototype,region_prototype
from .evaluate_sparse import summarize
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import fewclick_inputs
from benchmarks.v5.fair_comparison.metrics import score,VERSION
from benchmarks.v5.fair_comparison.protocol import nested

METHODS=('sam_ft','fewclick_m3d_image','fewclick_m3d_xcell')


class InferenceData(TransferData):
    def labels(self,*args,**kwargs):raise RuntimeError('GT access forbidden in adapted transfer inference')
    def supervision(self,*args,**kwargs):raise RuntimeError('query supervision forbidden in adapted transfer inference')


def run(tasks,development,training,method,output,gate=False,verified_gate=None):
    output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);device=torch.device('cpu');handles={}
    paths=[Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('prototypes.py'),
        Path(__file__).with_name('evaluate_sparse.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),
        Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/paper_multiscale/model.py')]
    sources={str(p):digest(p) for p in paths};protocol_sha=digest(tasks/'contract.json')
    try:
        completed=json.loads((training/'summary.json').read_text());selected=json.loads((training/'best_checkpoint.json').read_text())
        expected='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe'
        if completed['status']!='complete' or completed['step']!=10000 or not completed['full_manifest_parity']:
            raise ValueError('local adaptation incomplete')
        if completed['training_consumed_sha256']!=expected or digest(training/'training_consumed.jsonl')!=expected:
            raise ValueError('unmatched local training tasks')
        if digest(selected['path'])!=selected['checkpoint_sha256']:raise ValueError('selected development weight changed')
        if not gate:
            if verified_gate is None:raise ValueError('actual same-checkpoint transfer gate required')
            result=json.loads((verified_gate/'summary.json').read_text());meta=json.loads((verified_gate/'run.json').read_text())
            if result['status']!='complete' or not result['gate'] or result['rows_per_method']!=32:
                raise ValueError('actual adapted transfer gate incomplete')
            if meta['source_sha256']!=sources or meta['protocol_sha256']!=protocol_sha or meta['checkpoint_sha256']!=selected['checkpoint_sha256'] or meta['adapted_method']!=method:
                raise ValueError('transfer gate source/protocol/weight mismatch')
        scorer=TransferData(tasks);data=InferenceData(tasks)
        for row in data.rows.values():
            row['labels']='/unavailable/source_and_query_GT.png'
            for key in ('counts_json','interior_counts_json','eligible_json','labels_array_sha256'):row.pop(key,None)
        model=load(method,Path(selected['path']),device,digest(development/'contract.json'))
        prior=dict(model.prior)
        if method=='fewclick_m3d_xcell':prior.update(XCell_overlap='user-confirmed internal training, not current CRC/TIGER')
        plans=data.plans
        if gate:
            plans=[next(p for p in plans if p['dataset']==domain and p['kind']==kind and p['prompt_group']==0)
                for domain in sorted({p['dataset'] for p in plans}) for kind in ('same_present','cross_present','same_absent','cross_absent')]
        if method=='sam_ft':
            names=['sam_ft_feature_prototype']
            def encode_raw(raw):
                rgb=raw['rgb'][0].permute(1,2,0).numpy();transformed=model.transform.apply_image(rgb)
                image=torch.as_tensor(transformed).permute(2,0,1)
                return model.sam.image_encoder(model.sam.preprocess(image)[None])
        else:
            names=[method+'_graph_dense',method+'_same_feature_prototype']
            def encode_raw(raw):return model.core.encode_image(fewclick_inputs(raw))
        @lru_cache(maxsize=64)
        def encode(identity):return encode_raw(data.inputs(identity,model.contexts,model.cells))
        metadata=dict(worker_pid=os.getpid(),gate=gate,adapted_method=method,device='cpu',precision='float32',
            protocol_sha256=protocol_sha,development_protocol_sha256=digest(development/'contract.json'),
            checkpoint=selected['path'],checkpoint_sha256=selected['checkpoint_sha256'],source_sha256=sources,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initialization=prior,
            local_training_steps=10000,local_training_actual_sha256=expected,remote_training_steps=0,
            metric_version=VERSION,threshold=.5,prototype_temperature=.1,prototype_margin=0,query_human_clicks=0,
            query_class_ID_in_model=False,source_and_query_GT_in_input_object=False,cache_size=64,
            scope='local-adapted transfer generalization; same-feature prototypes are fixed-feature diagnostics, not retrained architecture ablations',
            absent_scope='no target annotated in valid query pixels; ignored pixels are not certified negative',
            latency_scope='joint learned decode+prototype for FewClick, feature retrieval for SAM; excludes encode/IO/scoring; not G8 efficiency')
        write(output/'run.json',metadata)
        folders={name:output/name for name in names};records={name:[] for name in names};checked=set();native_errors=[];coverage={}
        for name,folder in folders.items():
            folder.mkdir();write(folder/'run.json',metadata|dict(method=name));handles[name]=(folder/'rows.jsonl').open('x')
        with torch.inference_mode():
            for maximum in plans:
                source=encode(maximum['support_id']);query=encode(maximum['query_id'])
                for budget in scorer.contract['report_budgets']:
                    plan=nested(maximum,budget);value=data.materialize_inputs(plan,device,model.contexts,model.cells);prompts=value['prompts']
                    tick=time.perf_counter()
                    if method=='sam_ft':outputs={names[0]:dense_prototype(source,query,prompts)}
                    else:
                        task=model.core.support_task(source,prompts);native,maps=model.core.decode(query,task,prompts,False)
                        if maps.count_nonzero():raise ValueError('source coordinates rendered into remote query')
                        outputs={names[0]:native,names[1]:region_prototype(source,query,prompts)}
                    elapsed=1000*(time.perf_counter()-tick)
                    if gate and budget=='3+2' and plan['dataset'] not in checked:
                        if method=='sam_ft':
                            direct=dense_prototype(encode_raw(value['support']),encode_raw(value['query']),prompts)
                        else:
                            direct_result=model.core(fewclick_inputs(value['support']),fewclick_inputs(value['query']),prompts,False,False,with_support=False)
                            if direct_result['query_prompt_maps'].count_nonzero():raise ValueError('native remote query maps not zero')
                            direct=direct_result['query']
                        error=float((outputs[names[0]]['pixel_probability']-direct['pixel_probability']).abs().max())
                        if error!=0:raise ValueError('cached/direct adapted remote inference differs')
                        native_errors.append(error);checked.add(plan['dataset'])
                    # This independent scorer is the first object to access query GT.
                    truth=scorer.supervision(plan);qrow=scorer.rows[plan['query_id']]
                    coverage[plan['query_id']]=dict(valid_pixels=int(truth['valid'].sum()),total_pixels=int(truth['valid'].size))
                    for name,result in outputs.items():
                        probability=result['pixel_probability'][0].float().numpy()
                        if probability.shape!=(512,512) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                            raise ValueError('invalid adapted transfer probability')
                        metrics=score(probability>=.5,truth['truth'],truth['valid'],truth['mpp'])
                        if gate:
                            path=folders[name]/f'prediction_{len(records[name]):04d}.npy';np.save(path,probability)
                            pred=np.load(path,allow_pickle=False)>=.5
                            direct_counts=dict(tp=int((pred&truth['truth']&truth['valid']).sum()),fp=int((pred&~truth['truth']&truth['valid']).sum()),fn=int((~pred&truth['truth']&truth['valid']).sum()))
                            if any(metrics[k]!=v for k,v in direct_counts.items()):raise ValueError('adapted sparse saved score replay mismatch')
                        row=plan|metrics|dict(method=name,latency_ms=elapsed,query_human_clicks=0,
                            space_key=qrow['space_key'],source_id=qrow['source_id'],spatial_group_id=qrow['spatial_group_id'])
                        records[name].append(row);handles[name].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                if len(records[names[0]])%100==0 or gate:
                    for handle in handles.values():handle.flush()
                    state=dict(status='running',worker_pid=os.getpid(),rows_per_method=len(records[names[0]]),total_rows=len(plans)*4,
                        cache=encode.cache_info()._asdict(),seconds=time.time()-started)
                    write(output/'status.json',state);print(json.dumps(state),flush=True)
        for name,folder in folders.items():
            handles[name].close()
            write(folder/'summary.json',dict(status='complete',method=name,gate=gate,rows=len(records[name]),
                rows_sha256=digest(folder/'rows.jsonl'),protocol_sha256=protocol_sha,metrics=summarize(records[name]),scope=metadata['scope']))
        write(output/'query_annotation_coverage.json',coverage)
        write(output/'summary.json',dict(status='complete',worker_pid=os.getpid(),gate=gate,adapted_method=method,
            outputs={name:str(folder) for name,folder in folders.items()},rows_per_method=len(plans)*4,
            GT_forbidden_input_API=True,query_maps_zero=True if method!='sam_ft' else None,
            native_cached_errors=native_errors,saved_count_replay=gate,cache=encode.cache_info()._asdict(),seconds=time.time()-started))
        write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows_per_method=len(plans)*4))
    except Exception:
        for handle in handles.values():
            if not handle.closed:handle.close()
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path)
    a=p.parse_args();run(a.tasks,a.development,a.training,a.method,a.output,a.gate,a.verified_gate)
