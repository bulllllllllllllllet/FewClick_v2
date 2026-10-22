"""Strict replay of a saved multi-domain checkpoint on fixed prompt tasks.

This is an acceptance wrapper around the existing loader/evaluator; it does
not alter construction, prompts, thresholds, data paths or model parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips  # Load libvips before torch, matching the original pipeline.
import torch

from benchmarks.v5.multidomain_data.prepare import dump
from .episodes import EpisodeCorpus
from .evaluate import evaluate
from .model import load_model


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog','checkpoint','plans','reference-rows','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        torch.set_num_threads(args.threads)
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the original precision contract')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        source=checkpoint['run']
        if checkpoint['mapper']!='pixel':raise ValueError('expected the trained pixel prompt mapper')
        if file_sha(args.catalog/'tiles.parquet')!=source['catalog_sha256']:raise ValueError('catalog SHA changed; restore canonical paths instead of silently rewriting it')
        plans=json.loads(args.plans.read_text());reference=json.loads(args.reference_rows.read_text())
        key=lambda row:(row['number'],row['quality_condition'])
        references={key(row):row for row in reference}
        if len(references)!=len(reference) or len({key(p) for p in plans})!=len(plans) or not plans:raise ValueError('empty or duplicated task identities')
        for plan in plans:
            row=references[key(plan)]
            if any(row[k]!=value for k,value in plan.items()):raise ValueError('fixed task or click coordinates differ from reference')
        corpus=EpisodeCorpus(args.catalog,source['args']['seed'],prompt_policy=source['args']['prompt_policy'])
        model,weights=load_model(torch.device('cuda:0'))
        if weights!=source['weights']:raise ValueError('initialization identity differs from original run')
        model.load_state_dict(checkpoint['model'],strict=True);model.eval().requires_grad_(False)
        step=int(checkpoint['step']);amp=not source['args']['no_amp'];del checkpoint
        dump(args.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            wrapper_sha256=file_sha(__file__),checkpoint=str(args.checkpoint),checkpoint_sha256=file_sha(args.checkpoint),step=step,
            catalog=str(args.catalog),plans=str(args.plans),plans_sha256=file_sha(args.plans),reference_rows=str(args.reference_rows),
            reference_sha256=file_sha(args.reference_rows),device=torch.cuda.get_device_name(0),amp=amp,threshold=.5,query_GT_in_model=False))
        dump(args.output/'status.json',dict(status='evaluating',worker_pid=os.getpid(),tasks=len(plans)))
        report=evaluate(model,corpus,plans,torch.device('cuda:0'),args.output/'evaluation',amp=amp)
        rows=json.loads((args.output/'evaluation'/'rows.json').read_text());mismatches=[]
        for row in rows:
            original=references[key(row)]
            counts={name:dict(expected=original[name],actual=row[name]) for name in ('tp','fp','fn','positive','valid') if original[name]!=row[name]}
            scores={}
            for name in ('dice','bf1_um2','false_positive_fraction'):
                expected,actual=original[name],row[name]
                if (expected is None)!=(actual is None) or expected is not None and abs(expected-actual)>1e-7:
                    scores[name]=dict(expected=expected,actual=actual)
            if counts or scores:mismatches.append(dict(number=row['number'],quality=row['quality_condition'],counts=counts,scores=scores))
        verification=dict(status='passed' if not mismatches else 'failed',step=step,tasks=len(rows),exact_count_replay=not mismatches,
            mismatches=mismatches,metrics=report['overall'],seconds=time.time()-started,puma_used=False,test_used=False)
        dump(args.output/'verification.json',verification)
        if mismatches:raise ValueError('prediction replay differs; inspect verification.json, do not silently change clicks or threshold')
        dump(args.output/'status.json',dict(status='complete',worker_pid=os.getpid(),tasks=len(rows),seconds=time.time()-started))
        print('REPLAY_PASSED',json.dumps(verification),flush=True)
    except Exception as exc:
        dump(args.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)))
        raise


if __name__=='__main__':main()
