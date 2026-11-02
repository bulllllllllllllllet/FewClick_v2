"""Replay all functional masks and subgroup rules independently of the trainer."""
import argparse
import json
from pathlib import Path
import numpy as np
from .repaired_model import RepairedData
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'report.json').read_text());contract=json.loads((a.run/'contract.json').read_text())
    assert r['status']=='accepted_repaired_functional_gate' and r['exact_optimizer_continuation'] and r['fresh_final_predictions_exact']
    assert r['arm']==contract['arm'] and r['seed']==contract['seed'] and r['checkpoint_sha256']==sha(a.run/'checkpoint.pt')
    assert len(r['rows'])==len(contract['tasks'])==32
    data=RepairedData(a.cache);assert data.identity==contract['cache']
    with np.load(a.run/'masks.npz') as f:masks=f['masks']
    assert masks.shape==(32,32768)
    for row,task,packed in zip(r['rows'],contract['tasks'],masks):
        assert row['plan']==task and data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
        gt,mpp=data.labels(task['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool)
        assert score(mask,gt==task['target'],gt!=255,mpp)==row['metrics']
    for key,group in r['groups'].items():
        d,kind=key.split('|');rows=[x for x in r['rows'] if x['plan']['dataset']==d and x['plan']['kind']==kind];assert len(rows)==group['n'] and group['passed']
        if kind=='cross_absent':assert np.mean([x['metrics']['false_positive_fraction'] for x in rows])==group['values']['FPA']<=.1
        else:
            assert np.mean([x['metrics']['dice'] for x in rows])==group['values']['Dice']>=.2
            assert np.mean([x['valid_spatial_std'] for x in rows])==group['values']['spatial_std']>=.02
            assert np.median([x['swap_valid_mean'] for x in rows])==group['values']['median_swap_mean']>=.001
    receipt=dict(status='accepted_P3_spatial_train_learning_gate',arm=r['arm'],seed=r['seed'],masks_replayed=32,
                 report_sha256=sha(a.run/'report.json'),masks_sha256=sha(a.run/'masks.npz'),contract_sha256=sha(a.run/'contract.json'),
                 checkpoint_sha256=r['checkpoint_sha256'],acceptance_source_sha256=sha(__file__),formal_training_allowed=False)
    (a.run/'independent_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
