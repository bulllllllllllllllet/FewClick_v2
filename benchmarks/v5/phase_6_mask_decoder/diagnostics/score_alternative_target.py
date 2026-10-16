"""CPU-only rescoring of saved prompt-control masks against the new target GT."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import pyvips
import numpy as np
import pandas as pd
import torch
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch
from . import module_source
from .evaluate_dense_transfer import ROOT, DENSE_RUN, sha256
from .probe_boundary_refinement import metrics, summarize

SOURCE = ROOT/'support_query/prompt_dependence_1016_152335/formal_1016_152335'
CONDITIONS = ('correct', 'wrong', 'zero_task')


def completed_pairs(manifest, completed):
    if len(completed) != len(manifest['pairs']):
        raise ValueError('source incomplete')
    selected=[]
    for i,(pair,done) in enumerate(zip(manifest['pairs'],completed)):
        if any(done[k]!=v for k,v in pair.items()):raise ValueError('source identity changed')
        if done['wrong_index'] is not None:
            # done.number is inherited from the older 3841-task run. The
            # immediate source saved masks by its own 1946-pair loop index.
            selected.append({**done,'number':i})
    return selected


def score_saved(labels, saved, original_class, alternative_class):
    if original_class == alternative_class:
        raise ValueError('alternative target must differ')
    valid = labels != 255
    if not np.array_equal(valid, saved['valid']) or not np.array_equal(labels == original_class, saved['truth']):
        raise ValueError('decoded GT differs from original saved truth/valid')
    truth = (labels == alternative_class) & valid
    present = bool(truth.any()); results = []
    for condition in CONDITIONS:
        for model in ('j5', 'dense'):
            pred = saved[condition+'_'+model]
            if pred.shape != labels.shape or pred.dtype != np.bool_:
                raise ValueError('saved prediction shape/type')
            p, t, v = [torch.from_numpy(x.copy())[None] for x in (pred, truth, valid)]
            row = {'condition':condition,'model':model,'alternative_present':present,
                   'alternative_pixels':int(truth.sum()),'valid_pixels':int(valid.sum())}
            if present:
                row.update(metrics(p,t,v))
                if condition == 'wrong':
                    old = (labels == original_class) & valid
                    tp = int((pred & old).sum()); fp = int((pred & ~old & valid).sum()); fn = int((~pred & old).sum())
                    row['original_target_dice'] = 2*tp/max(2*tp+fp+fn,1)
                    row['alternative_minus_original_dice'] = row['dice']-row['original_target_dice']
            else:
                area = int((pred & valid).sum())/max(int(valid.sum()),1)
                row.update(false_positive_area=area,any_activation=bool((pred & valid).any()),area_activation_1pct=area>=.01)
            results.append(row)
    return results


def summarize_alternative(rows):
    output = {}
    for protocol in sorted({r['protocol'] for r in rows}):
        output[protocol] = {}
        for model in ('j5','dense'):
            rr = [r for r in rows if r['protocol']==protocol and r['model']==model]
            groups = {}
            for present in (True,False):
                subset = [r for r in rr if r['alternative_present']==present]
                results = {}
                for condition in CONDITIONS:
                    cc = [r for r in subset if r['condition']==condition]
                    if not cc:
                        results[condition] = {'episodes':0};continue
                    if present:
                        stats = summarize([r | {'prompt_violations':0} for r in cc]);stats.pop('prompt_violations')
                        stats['per_alternative_class'] = {str(c):{'episodes':len(cr),'dice':float(np.mean([r['dice'] for r in cr]))}
                            for c in sorted({r['wrong_target_class'] for r in cc})
                            for cr in [[r for r in cc if r['wrong_target_class']==c]]}
                        if condition == 'wrong':
                            correct = {r['number']:r for r in subset if r['condition']=='correct'}
                            blocks = sorted({r['query_spatial_group'] for r in cc})
                            deltas = [[r['dice']-correct[r['number']]['dice'] for r in cc if r['query_spatial_group']==b] for b in blocks]
                            sums=np.array([sum(v) for v in deltas]);counts=np.array([len(v) for v in deltas])
                            rng=np.random.default_rng(20261016);draws=rng.integers(0,len(blocks),(2000,len(blocks)))
                            boot=sums[draws].sum(1)/counts[draws].sum(1)
                            stats['new_prompt_minus_original_prompt_dice']={'mean':float(sums.sum()/counts.sum()),
                                'spatial_blocks':len(blocks),'spatial_block_bootstrap_95':np.percentile(boot,[2.5,97.5]).tolist()}
                            stats['original_target_dice'] = float(np.mean([r['original_target_dice'] for r in cc]))
                            stats['fraction_preferring_alternative_gt'] = float(np.mean([r['alternative_minus_original_dice']>0 for r in cc]))
                    else:
                        stats={'episodes':len(cc),**{k:float(np.mean([r[k] for r in cc])) for k in
                            ('false_positive_area','any_activation','area_activation_1pct')}}
                    results[condition]=stats
                groups['present' if present else 'absent']=results
            output[protocol][model]=groups
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if ROOT not in args.output.resolve().parents:raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'labels').mkdir()
    torch.set_num_threads(4);started=time.time()
    source_manifest=json.loads((SOURCE/'run.json').read_text())
    done=[json.loads(x) for x in (SOURCE/'completed.jsonl').read_text().splitlines()]
    if len(done)!=len(source_manifest['pairs']) or json.loads((SOURCE/'status.json').read_text())['status']!='complete':
        raise ValueError('source incomplete')
    selected=completed_pairs(source_manifest,done)
    if args.smoke:
        unique={}
        for p in selected:unique.setdefault((p['protocol'],p['wrong_target_class']),p)
        selected=list(unique.values())
    training=json.loads((DENSE_RUN/'run.json').read_text())
    meta=json.loads((Path(training['checkpoint']).parent/'run_metadata.json').read_text())
    class_map=load_config(Path(meta['inputs']['phase2_config']))['data']['class_map']
    if int(meta['config']['data']['ignore_index'])!=255:raise ValueError('unsupported ignore')
    patches_path=Path(training['args']['spatial_data'])/'patches_index.parquet'
    frame=pd.read_parquet(patches_path).set_index('patch_id')
    manifest={'source':str(SOURCE),'source_revision':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        'source_sha256':sha256(module_source('score_alternative_target')),'source_manifest_sha256':sha256(SOURCE/'run.json'),
        'source_completed_sha256':sha256(SOURCE/'completed.jsonl'),'patch_index':str(patches_path),'patch_index_sha256':sha256(patches_path),
        'class_map':class_map,'pairs':selected,'train':False,'test_used':False,'gpu_used':False,'smoke':args.smoke,
        'scope':source_manifest['scope'],'worker_pid':os.getpid(),'note':'Binary saved masks scored at original threshold; no new model inference'}
    (args.output/'run.json').write_text(json.dumps(manifest,indent=2))
    rows=[];label_files={}
    for i,pair in enumerate(selected):
        pid=pair['query_id'];record=frame.loc[pid]
        if record['split']!='val':raise ValueError('query outside val')
        if pid not in label_files:
            labels=decode_gt_patch(Path(record.gt_path),int(record.x_10x),int(record.y_10x),int(record.width_10x),int(record.height_10x),class_map,255)
            file=args.output/'labels'/f'{len(label_files):05d}.npz';np.savez_compressed(file,labels=labels)
            label_files[pid]=str(file)
        else:
            with np.load(label_files[pid]) as cached:labels=cached['labels']
        original_mask=SOURCE/'masks'/f"{pair['number']:05d}.npz"
        with np.load(original_mask) as saved:scored=score_saved(labels,saved,pair['target_class'],pair['wrong_target_class'])
        for r in scored:
            r.update({k:pair[k] for k in ('number','source_number','protocol','query_id','support_id','target_class','wrong_target_class','query_spatial_group')})
            r.update(source_mask=str(original_mask),source_mask_sha256=sha256(original_mask),label_file=label_files[pid],
                     label_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
            rows.append(r)
        with (args.output/'metrics.jsonl').open('a') as f:
            for r in scored:f.write(json.dumps(r)+'\n')
        with (args.output/'completed.jsonl').open('a') as f:f.write(json.dumps({'number':pair['number'],'query_id':pid,'alternative_present':scored[0]['alternative_present']})+'\n')
        (args.output/'status.json').write_text(json.dumps({'status':'scoring','worker_pid':os.getpid(),'completed':i+1,'pairs':len(selected),'elapsed_seconds':time.time()-started}))
        print('PAIR',i+1,'/',len(selected),'present',scored[0]['alternative_present'],'seconds',round(time.time()-started,1),flush=True)
    report={'summary':summarize_alternative(rows),'pairs':len(selected),'pair_counts':dict(Counter(p['protocol'] for p in selected)),
        'original_gt_valid_exact':True,'unique_query_labels':len(label_files),'elapsed_seconds':time.time()-started,'scope':manifest['scope'],'smoke':args.smoke}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid(),'completed':len(selected)}))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
