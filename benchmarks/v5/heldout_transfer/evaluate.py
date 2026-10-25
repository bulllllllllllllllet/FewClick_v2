"""Seen-selected heldout checkpoints on frozen remote query tasks, CPU only."""
import argparse
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
import torch

from .features import sam_features,wsi_features
from benchmarks.v5.heldout_comparison.audit import FOLDS
from benchmarks.v5.heldout_comparison.evaluate import binding
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.local_data import fewclick_inputs
from benchmarks.v5.fair_comparison.metrics import score,VERSION
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.transfer_comparison.data import TransferData
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype,region_prototype
from benchmarks.v5.transfer_comparison.evaluate_sparse import summarize

METHODS=('fewclick_m3d_image','sam_ft','wsi_sam_ft')


def read(path):return json.loads(Path(path).read_text())


class InferenceData(TransferData):
    def __init__(self,tasks):
        super().__init__(tasks)
        for row in self.rows.values():
            row['labels']='/unavailable/inference_GT_forbidden.png'
            for key in ('counts_json','interior_counts_json','eligible_json','labels_array_sha256'):row.pop(key,None)
    def labels(self,*args,**kwargs):raise RuntimeError('Inference GT access forbidden')
    def supervision(self,*args,**kwargs):raise RuntimeError('Inference supervision forbidden')


def sources():
    files=list(Path(__file__).parent.glob('*.py'))
    files += [Path('benchmarks/v5/heldout_comparison/evaluate.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),
        Path('benchmarks/v5/transfer_comparison/data.py'),Path('benchmarks/v5/transfer_comparison/prototypes.py'),
        Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/paper_multiscale/model.py')]
    return {str(p):digest(p) for p in files}


def validate(a):
    if a.checkpoint_gate and not a.gate:raise ValueError('small checkpoint allowed only for integration gate')
    identity=binding(a.training_tasks,a.training,a.local_test,a.method,gate=a.checkpoint_gate)
    remote=read(a.tasks/'contract.json');fold=identity['fold']
    if (remote.get('heldout_fold')!=fold or remote.get('checkpoint_selection_allowed') is not False or
        remote.get('training_protocol_sha256')!=identity['training_protocol_sha256'] or
        Path(remote['training_protocol']).resolve()!=a.training_tasks.resolve()):
        raise ValueError('heldout remote/training binding mismatch')
    coords=read(a.realized/'summary.json');sha=digest(a.realized/'training_plans.jsonl')
    if coords['training_plans_sha256']!=sha or identity['training_contract']['training_plans_sha256']!=sha:
        raise ValueError('actual training coordinate identity mismatch')
    if not a.checkpoint_gate:
        done=read(a.training/'summary.json')
        if not coords['complete_protocol'] or coords['tasks']!=40000 or done['training_consumed_sha256']!=sha or digest(a.training/'training_consumed.jsonl')!=sha:
            raise ValueError('formal full actual training consumption mismatch')
    plans=read(a.tasks/'validation_maximum.json')
    if digest(a.tasks/'validation_maximum.json')!=remote['tasks_sha256'] or any(p['target'] not in FOLDS[fold]['test'][p['dataset']] for p in plans):
        raise ValueError('remote heldout test task identity/membership')
    return identity,sha,plans


def run(a):
    if torch.cuda.is_available():raise ValueError('CPU remote evaluation must hide CUDA')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time();handles={}
    try:
        identity,actual_sha,plans=validate(a);source=sources();protocol_sha=digest(a.tasks/'contract.json')
        if not a.gate:
            if a.verified_gate is None:raise ValueError('same-checkpoint actual remote gate required')
            accepted=read(a.verified_gate/'summary.json');prior=read(a.verified_gate/'run.json')
            if (accepted['status']!='complete' or not accepted['gate'] or prior['checkpoint_gate'] or
                prior['checkpoint_sha256']!=identity['checkpoint_sha256'] or prior['source_sha256']!=source or
                prior['protocol_sha256']!=protocol_sha or prior['adapted_method']!=a.method):
                raise ValueError('formal remote gate identity mismatch')
        scorer=TransferData(a.tasks);data=InferenceData(a.tasks)
        for forbidden in (data.labels,data.supervision):
            try:forbidden('probe')
            except RuntimeError:pass
            else:raise ValueError('GT input API unexpectedly enabled')
        model=load(a.method,Path(identity['checkpoint']),'cpu',identity['training_protocol_sha256'])
        if a.method=='sam_ft':
            names=['sam_ft_feature_prototype'];encode_raw=lambda raw:sam_features(model,raw)
        elif a.method=='wsi_sam_ft':
            names=['wsi_sam_ft_two_view_feature_prototype'];encode_raw=lambda raw:wsi_features(model,raw)
        else:
            names=['fewclick_m3d_image_graph_dense','fewclick_m3d_image_same_feature_prototype']
            encode_raw=lambda raw:model.core.encode_image(fewclick_inputs(raw))
        pair_plans={p['pair_id']:p for p in plans}
        coverage=dict(pairs=len(pair_plans),maximum_point_groups=len(plans),
            pair_coverage=dict(Counter(p['dataset']+'|'+p['kind'] for p in pair_plans.values())),
            unique_support_fields=len({p['support_id'] for p in plans}),unique_query_fields=len({p['query_id'] for p in plans}))
        if a.gate:
            strata=sorted({(p['dataset'],p['kind']) for p in plans})
            plans=[next(p for p in plans if (p['dataset'],p['kind'])==s and p['prompt_group']==0) for s in strata]
        @lru_cache(maxsize=64)
        def encode(identity):return encode_raw(data.inputs(identity,model.contexts,model.cells))
        metadata=dict(worker_pid=os.getpid(),gate=a.gate,checkpoint_gate=a.checkpoint_gate,adapted_method=a.method,
            checkpoint_sha256=identity['checkpoint_sha256'],checkpoint=identity['checkpoint'],training_protocol_sha256=identity['training_protocol_sha256'],
            actual_training_plans_sha256=actual_sha,protocol_sha256=protocol_sha,source_sha256=source,metric_version=VERSION,
            actual_subset_coverage=coverage,parent_count_fields='Inherited parent counts in test contract are ancestry only; actual subset recounted here',
            query_human_clicks=0,query_class_ID_in_model=False,GT_forbidden_input_API=True,remote_supervised_updates=0,
            threshold=.5,prototype_temperature=.1,initialization=model.prior,
            scope='Heldout-label local adaptation -> sparse support/query generalization; single seed, not all-pretraining-unseen',
            prototype_scope='Custom frozen encoder retrieval wrappers, not official SAM/WSI-SAM native remote decoders',
            absent_scope='No target annotated within valid query pixels; unknown pixels not certified negatives',
            latency_scope='Decode/prototype only; excludes encoding/IO/scoring; not WSI efficiency')
        write(a.output/'run.json',metadata);folders={n:a.output/n for n in names};records={n:[] for n in names};errors=[];annotation={}
        for name,folder in folders.items():folder.mkdir();write(folder/'run.json',metadata|dict(method=name));handles[name]=(folder/'rows.jsonl').open('x')
        with torch.inference_mode():
            for maximum in plans:
                source_encoded=encode(maximum['support_id']);query_encoded=encode(maximum['query_id'])
                for budget in scorer.contract['report_budgets']:
                    plan=nested(maximum,budget);value=data.materialize_inputs(plan,'cpu',model.contexts,model.cells);prompts=value['prompts'];tick=time.perf_counter()
                    if a.method=='fewclick_m3d_image':
                        task=model.core.support_task(source_encoded,prompts);native,maps=model.core.decode(query_encoded,task,prompts,False)
                        if maps.count_nonzero():raise ValueError('remote query received source-coordinate maps')
                        outputs={names[0]:native,names[1]:region_prototype(source_encoded,query_encoded,prompts)}
                    else:outputs={names[0]:dense_prototype(source_encoded,query_encoded,prompts)}
                    elapsed=1000*(time.perf_counter()-tick)
                    if a.gate and budget=='3+2':
                        if a.method=='fewclick_m3d_image':
                            direct=model.core(fewclick_inputs(value['support']),fewclick_inputs(value['query']),prompts,False,False,with_support=False)
                            if direct['query_prompt_maps'].count_nonzero():raise ValueError('native remote query maps nonzero')
                            expected=direct['query']
                        else:expected=dense_prototype(encode_raw(value['support']),encode_raw(value['query']),prompts)
                        error=float((expected['pixel_probability']-outputs[names[0]]['pixel_probability']).abs().max());errors.append(error)
                        if error!=0:raise ValueError('native/cached remote prediction differs')
                    truth=scorer.supervision(plan);qrow=scorer.rows[plan['query_id']]
                    annotation[plan['query_id']]=dict(valid_pixels=int(truth['valid'].sum()),total_pixels=int(truth['valid'].size))
                    for name,result in outputs.items():
                        p=result['pixel_probability'][0].float().numpy()
                        if p.shape!=(512,512) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('invalid remote probability')
                        metrics=score(p>=.5,truth['truth'],truth['valid'],truth['mpp'])
                        if a.gate:
                            path=folders[name]/f'prediction_{len(records[name]):04d}.npy';np.save(path,p);pred=np.load(path,allow_pickle=False)>=.5
                            direct=dict(tp=int((pred&truth['truth']&truth['valid']).sum()),fp=int((pred&~truth['truth']&truth['valid']).sum()),fn=int((~pred&truth['truth']&truth['valid']).sum()))
                            if any(metrics[k]!=v for k,v in direct.items()):raise ValueError('saved gate probability/count mismatch')
                        row=plan|metrics|dict(method=name,latency_ms=elapsed,query_human_clicks=0,space_key=qrow['space_key'],source_id=qrow['source_id'],spatial_group_id=qrow['spatial_group_id'])
                        records[name].append(row);handles[name].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                for handle in handles.values():handle.flush()
                write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),rows_per_method=len(records[names[0]]),total_rows=4*len(plans),seconds=time.time()-started))
        for name,folder in folders.items():
            handles[name].close();write(folder/'summary.json',dict(status='complete',gate=a.gate,method=name,rows=len(records[name]),metrics=summarize(records[name]),rows_sha256=digest(folder/'rows.jsonl')))
        write(a.output/'query_annotation_coverage.json',annotation)
        write(a.output/'summary.json',dict(status='complete',gate=a.gate,checkpoint_gate=a.checkpoint_gate,worker_pid=os.getpid(),
            outputs={n:str(p) for n,p in folders.items()},rows_per_method=4*len(plans),native_cached_errors=errors,
            GT_forbidden_input_API=True,query_maps_zero=True if a.method=='fewclick_m3d_image' else None,
            saved_count_replay=a.gate,seconds=time.time()-started))
        write(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows_per_method=4*len(plans)))
    except Exception:
        for handle in handles.values():
            if not handle.closed:handle.close()
        write(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('tasks','training-tasks','local-test','training','realized','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--checkpoint-gate',action='store_true')
    p.add_argument('--verified-gate',type=Path);run(p.parse_args())
