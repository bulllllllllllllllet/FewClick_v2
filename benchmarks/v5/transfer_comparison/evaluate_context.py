"""Full native context evaluation with sparse-generated versus dense-reference tracks."""
import argparse
import hashlib
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
from .multiverseg import MultiverSegContext
from .evaluate_sparse import summarize
from benchmarks.v5.external_evaluation.evaluate_public import PublicModel
from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.fair_comparison.metrics import score,VERSION
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v4.baseline.adapters.base import EpisodeRequest


class InputsOnly(TransferData):
    def labels(self,*args,**kwargs):raise RuntimeError('tissue GT forbidden in context inference')
    def supervision(self,*args,**kwargs):raise RuntimeError('query supervision forbidden in context inference')


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def run(tasks,sources,weight,dependencies,output,gate=False,verified_gate=None):
    output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    source_files=[Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('multiverseg.py'),
        Path(__file__).with_name('evaluate_sparse.py'),Path('benchmarks/v5/external_evaluation/evaluate_public.py')]
    snapshot={str(p):digest(p) for p in source_files};protocol_sha=digest(tasks/'contract.json')
    handles={}
    try:
        scorer=TransferData(tasks);inference=InputsOnly(tasks)
        for row in inference.rows.values():
            row['labels']='/unavailable/source_and_query_tissue_GT.png'
            for key in ('counts_json','interior_counts_json','eligible_json','labels_array_sha256'):row.pop(key,None)
        # Qualification reads source GT only in the benchmark/scorer object.
        support_ids=sorted({p['support_id'] for p in scorer.plans})
        coverage={identity:dict(valid=int((scorer.labels(identity)!=255).sum()),total=int(scorer.labels(identity).size),
            labels_sha256=scorer.rows[identity]['labels_array_sha256']) for identity in support_ids}
        pairs={p['pair_id']:p for p in scorer.plans if p['prompt_group']==0}
        eligible={key:plan for key,plan in pairs.items() if coverage[plan['support_id']]['valid']==coverage[plan['support_id']]['total']}
        excluded=[dict(pair_id=key,support_id=plan['support_id'],reason='incomplete support annotation') for key,plan in pairs.items() if key not in eligible]
        write(output/'support_coverage.json',dict(fields=coverage,full_GT_eligible_pair_ids=sorted(eligible),excluded=excluded))
        if not gate:
            if verified_gate is None:raise ValueError('actual native context stream gate required')
            old=json.loads((verified_gate/'summary.json').read_text());meta=json.loads((verified_gate/'run.json').read_text())
            if old['status']!='complete' or not old['gate'] or old['rows']!={'predicted_sparse':16,'full_GT_reference':4}:
                raise ValueError('native full-path gate incomplete')
            if meta['source_sha256']!=snapshot or meta['protocol_sha256']!=protocol_sha:
                raise ValueError('gated context source/protocol changed')
            if meta['sources_sha256']!=digest(sources) or meta['weight_manifest_sha256']!=digest(weight):
                raise ValueError('gated official source/weight manifest changed')
        plans=scorer.plans
        if gate:
            plans=[next(p for p in scorer.plans if p['dataset']==d and p['query_presence']==presence and p['prompt_group']==0)
                for d in sorted({p['dataset'] for p in scorer.plans}) for presence in ('present','absent')]
            references=[next(p for p in eligible.values() if p['dataset']==d and p['query_presence']==presence)
                for d in sorted({p['dataset'] for p in scorer.plans}) for presence in ('present','absent')]
        else:references=list(eligible.values())
        # Traversal is reordered only to reuse source image embeddings, never by scores.
        plans=sorted(plans,key=lambda p:(p['support_id'],p['target'],p['prompt_group'],p['pair_id']))
        model=MultiverSegContext(sources,weight,dependencies,'cpu');generator=PublicModel('sam')
        if not gate and (meta['prior']!=model.prior or meta['generator']!=generator.prior):
            raise ValueError('gated weights/dependencies/generator provenance changed')
        metadata=dict(worker_pid=os.getpid(),gate=gate,device='cpu',threads=4,protocol_sha256=protocol_sha,
            source_sha256=snapshot,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            prior=model.prior,generator=generator.prior,metric_version=VERSION,query_human_clicks=0,query_class_ID_in_model=False,
            source_and_query_GT_forbidden_in_input_object=True,full_GT_scope='source only, separately qualified dense reference',
            sources_sha256=digest(sources),weight_manifest_sha256=digest(weight),
            temporal_scope='CPU with source-mask cache; no independent GPU/WSI efficiency claim')
        write(output/'run.json',metadata)
        folders={kind:output/kind for kind in ('predicted_sparse','full_GT_reference')};results={kind:[] for kind in folders}
        for kind,folder in folders.items():
            folder.mkdir();handles[kind]=(folder/'rows.jsonl').open('x')
            write(folder/'run.json',metadata|dict(method='multiverseg_'+kind,annotation_track=kind))
        cache_dir=output/'source_contexts';cache_dir.mkdir();contexts={};native_errors=[];checked_sources=set()
        def context(plan,source):
            variant=json.loads(inference.rows[plan['support_id']]['variants_json'])['original']
            key_record=dict(support_id=plan['support_id'],image_sha256=variant['image_sha256'],
                positive_xy=plan['positive_xy'],negative_xy=plan['negative_xy'],generator_weight_sha256=generator.config['checkpoint_sha256'])
            key=hashlib.sha256(json.dumps(key_record,sort_keys=True).encode()).hexdigest()
            if key in contexts:
                saved=contexts[key]
                if digest(saved['path'])!=saved['file_sha256']:raise ValueError('predicted source context changed')
                with np.load(saved['path'],allow_pickle=False) as arrays:mask=arrays['mask'].copy()
                if array_sha(mask)!=saved['mask_sha256']:raise ValueError('source context mask identity differs')
                return mask,saved
            rgb=source[0].permute(1,2,0).numpy();pos=np.asarray(plan['positive_xy'],np.float32)*512;neg=np.asarray(plan['negative_xy'],np.float32)*512
            probability,candidates=generator.predict(dict(fine=rgb),pos,neg,plan['support_id'])
            if probability.shape!=(512,512) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
                raise ValueError('invalid source generator probability')
            if gate and plan['support_id'] not in checked_sources:
                request=EpisodeRequest(str(plan['pair_id']),rgb,pos,neg,'point');request.validate()
                direct=generator.model.predict(request).probability
                error=float(np.max(np.abs(probability-direct)));native_errors.append(error);checked_sources.add(plan['support_id'])
                if error>1e-7 or not np.array_equal(probability>=.5,direct>=.5):raise ValueError('cached source generator differs from native SAM')
            mask=(probability>=.5).astype(np.uint8);path=cache_dir/f'{key}.npz'
            if path.exists():raise ValueError('unexpected existing context artifact')
            np.savez_compressed(path,mask=mask)
            item=key_record|dict(path=str(path),file_sha256=digest(path),mask_sha256=array_sha(mask),candidates=candidates)
            contexts[key]=item
            return mask,item
        def record(kind,plan,mask,mask_kind,source_valid=None,context_meta=None):
            value=inference.materialize_inputs(plan,torch.device('cpu'),False,False)
            tick=time.perf_counter()
            result=model.predict(value['support']['rgb'],value['query']['rgb'],mask,
                mask_kind=mask_kind,source_valid=source_valid,verify_native=gate)
            elapsed=1000*(time.perf_counter()-tick)
            probability=result['pixel_probability'][0].numpy()
            truth=scorer.supervision(plan);metric=score(probability>=.5,truth['truth'],truth['valid'],truth['mpp'])
            if gate:
                path=folders[kind]/f'prediction_{len(results[kind]):04d}.npy';np.save(path,probability)
                pred=np.load(path,allow_pickle=False)>=.5
                actual=dict(tp=int((pred&truth['truth']&truth['valid']).sum()),fp=int((pred&~truth['truth']&truth['valid']).sum()),fn=int((~pred&truth['truth']&truth['valid']).sum()))
                if any(metric[k]!=v for k,v in actual.items()):raise ValueError('context saved score replay differs')
            row=scorer.rows[plan['query_id']]
            output_plan=plan if kind=='predicted_sparse' else {k:v for k,v in plan.items() if k not in ('positive_xy','negative_xy','click_audit')}
            output_plan=output_plan|dict(click_budget='full_mask',prompt_group=-1) if kind=='full_GT_reference' else output_plan
            item=output_plan|metric|dict(method='multiverseg_'+kind,annotation_track=kind,mask_kind=mask_kind,
                support_human_points=len(plan['positive_xy'])+len(plan['negative_xy']) if kind=='predicted_sparse' else 0,
                support_dense_annotated_pixels=262144 if kind=='full_GT_reference' else 0,
                source_mask_sha256=array_sha(mask),source_context=context_meta,
                latency_ms=elapsed,query_human_clicks=0,space_key=row['space_key'],source_id=row['source_id'],spatial_group_id=row['spatial_group_id'])
            results[kind].append(item);handles[kind].write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');handles[kind].flush()
        for maximum in plans:
            value=inference.materialize_inputs(maximum,torch.device('cpu'),False,False)
            for budget in scorer.contract['report_budgets']:
                plan=nested(maximum,budget);mask,entry=context(plan,value['support']['rgb'])
                record('predicted_sparse',plan,mask,'model_prediction',context_meta=entry)
            if len(results['predicted_sparse'])%100==0 or gate:
                state=dict(status='running',worker_pid=os.getpid(),stage='predicted_sparse',completed=len(results['predicted_sparse']),total=len(plans)*4,source_contexts=len(contexts),seconds=time.time()-started)
                write(output/'status.json',state);print(json.dumps(state),flush=True)
        write(output/'source_context_index.json',contexts)
        for maximum in references:
            labels=scorer.labels(maximum['support_id']);mask=(labels==maximum['target']).astype(np.uint8)
            record('full_GT_reference',maximum,mask,'full_GT',source_valid=labels!=255)
            if len(results['full_GT_reference'])%50==0 or gate:
                state=dict(status='running',worker_pid=os.getpid(),stage='full_GT_reference',completed=len(results['full_GT_reference']),total=len(references),seconds=time.time()-started)
                write(output/'status.json',state);print(json.dumps(state),flush=True)
        for kind,folder in folders.items():
            handles[kind].close()
            expected=len(plans)*4 if kind=='predicted_sparse' else len(references)
            if len(results[kind])!=expected:raise ValueError('context track incomplete')
            write(folder/'summary.json',dict(status='complete',method='multiverseg_'+kind,gate=gate,rows=expected,
                protocol_sha256=protocol_sha,rows_sha256=digest(folder/'rows.jsonl'),metrics=summarize(results[kind]),
                qualification='all935 sparse pairs' if kind=='predicted_sparse' else 'only full-source-GT eligible pairs; compare other methods on same subset'))
        write(output/'summary.json',dict(status='complete',worker_pid=os.getpid(),gate=gate,
            rows={kind:len(values) for kind,values in results.items()},outputs={k:str(v) for k,v in folders.items()},
            source_contexts=len(contexts),native_SAM_errors=native_errors,GT_forbidden_input_object=True,
            native_context_forward_identical=gate,saved_count_replay=gate,
            full_GT_eligible_pairs=len(eligible),full_GT_excluded_pairs=len(excluded),seconds=time.time()-started))
        write(output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows={k:len(v) for k,v in results.items()}))
    except Exception:
        for handle in handles.values():
            if not handle.closed:handle.close()
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--weight',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path)
    a=p.parse_args();run(a.tasks,a.sources,a.weight,a.dependencies,a.output,a.gate,a.verified_gate)
