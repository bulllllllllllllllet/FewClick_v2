"""Apply the declared tissue-target scope to immutable saved predictions."""
import argparse
import json
import os
from pathlib import Path
import time

import pyvips
import numpy as np
from PIL import Image

from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN,ROOT,sha256
from .diagnose_regression import build_summary
from .evaluate import summarize as dynamic_summary
from .regression import dump,report
from .scoring import scoring_policy,split_target_tasks


def verify_negative_background(source_rows,shards):
    for row in source_rows:
        if row['weight']!='C3' or row['prompt']!='interior_pixel' or row['target_class']!=0:continue
        folders=[p/'cases'/f'{row["position"]:04d}' for p in shards]
        found=[p for p in folders if (p/'predictions.npz').exists()]
        if len(found)!=1:raise ValueError('unique saved mask required')
        folder=found[0];labels=np.asarray(Image.open(folder/'labels.png'))
        with np.load(folder/'predictions.npz') as saved:
            valid=saved['valid'];truth=saved['truth']&valid;pred=saved['C3|legacy|interior_pixel']&valid
            background=(labels==2)&valid
            if not background.any():continue
            if (truth&background).any():raise ValueError('background became positive in tumor task')
            counts=dict(tp=int((pred&truth).sum()),fp=int((pred&~truth&valid).sum()),fn=int((~pred&truth).sum()))
            if any(counts[k]!=row[k] for k in counts):raise ValueError('valid negative pixel counts changed')
            kept,excluded=split_target_tasks([row],dataset='crc')
            if kept!=[row] or excluded:raise ValueError('tissue task with negative background removed')
            return dict(position=row['position'],negative_background_pixels=int(background.sum()),
                        false_positive_background_pixels=int((pred&background).sum()),counts=counts,
                        background_pixels_kept_valid=True,mask_file=str(folder/'predictions.npz'))
    raise ValueError('no real tissue task with background pixels found')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--compatibility',type=Path,required=True)
    p.add_argument('--longrun',type=Path,required=True);p.add_argument('--mask-shards',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if ROOT not in a.output.resolve().parents:raise ValueError('artifact root')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        source_path=a.diagnosis/'rows.json';rows=json.loads(source_path.read_text());old=json.loads((a.diagnosis/'summary.json').read_text())
        result=build_summary(rows);primary,excluded=split_target_tasks(rows,dataset='crc')
        result.update(source_rows=str(source_path),source_rows_sha256=sha256(source_path),worker_pid=os.getpid(),
                      task_positions=len({r['position'] for r in rows}),primary_task_positions=len({r['position'] for r in primary}),
                      excluded_background_task_positions=len({r['position'] for r in excluded}),
                      model_weights_changed=False,predictions_changed=False,puma_used=False,test_used=False,
                      scope='User-defined tissue-target development score; same GT-interior clicks/old RGB/old cells; historical centroid protocols remain separate')
        for key,value in result['conditions'].items():
            if value['legacy_all_targets']!=old['conditions'][key]['overall']:raise ValueError('historical all-target aggregate changed')
            if value['overall']!=old['conditions'][key]['nonbackground_diagnostic']:raise ValueError('existing tissue subset not reproduced')
        sets={key:{r['position'] for r in primary if '|'.join([r['weight'],r['cells'],r['prompt']])==key} for key in result['conditions']}
        if any(v!=next(iter(sets.values())) for v in sets.values()):raise ValueError('models/protocols have different primary query sets')
        result['negative_background_verification']=verify_negative_background(rows,a.mask_shards)
        dump(a.output/'summary.json',result)
        compatible_rows=json.loads((a.compatibility/'rows.json').read_text());refs=json.loads((DENSE_RUN/'run.json').read_text())['refs']['val']
        dump(a.output/'compatibility_summary.json',dict(models=report(compatible_rows,refs),scoring_policy=scoring_policy(),
            source_rows_sha256=sha256(a.compatibility/'rows.json'),same_prompt_semantics=False))
        checks=[]
        for arm in json.loads((a.longrun/'long20k_selection.json').read_text()):
            folder=Path(arm['output']);best=json.loads((folder/'best_checkpoint.json').read_text());summary=json.loads((folder/'summary.json').read_text())
            path=folder/f'validation_step_{best["step"]:05d}'/'rows.json';saved=json.loads(path.read_text());new=dynamic_summary(saved)
            previous=next(h['metrics']['overall'] for h in summary['history'] if h['step']==best['step'])
            if previous!=new:raise ValueError('background-free dynamic validation unexpectedly changed')
            checks.append(dict(arm=arm['arm'],step=best['step'],rows_sha256=sha256(path),metrics=new,unchanged=True))
        dump(a.output/'dynamic_checks.json',checks)
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
