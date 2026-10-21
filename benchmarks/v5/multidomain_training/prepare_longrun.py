"""Lock extended development episodes without reading heldout PUMA/test."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import pyvips
import torch

from benchmarks.v5.multidomain_data.prepare import dump,sha
from .episodes import EpisodeCorpus
from .longrun import extend_validation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--anchor-plans',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4)
    try:
        anchors=json.loads(a.anchor_plans.read_text());corpus=EpisodeCorpus(a.catalog);plans=extend_validation(corpus,anchors)
        if plans[:len(anchors)]!=anchors:raise ValueError('original validation anchor changed')
        if len({(p['number'],p['quality_condition']) for p in plans})!=len(plans):raise ValueError('duplicate validation task identity')
        for i,plan in enumerate(plans):
            if plan['split']!='val':raise ValueError('development validation must stay in val')
            corpus.materialize(plan,torch.device('cpu'))
            if (i+1)%100==0:dump(a.output/'status.json',dict(status='checking',worker_pid=os.getpid(),completed=i+1,total=len(plans)))
        dump(a.output/'validation_plans.json',plans)
        dump(a.output/'summary.json',dict(status='complete',worker_pid=os.getpid(),episodes=len(plans),anchor_episodes=len(anchors),
            counts=dict(Counter(p['dataset']+'|'+p['kind'] for p in plans)),anchor_unchanged=True,
            source_anchor_sha256=sha(a.anchor_plans.read_bytes()),plans_sha256=sha((a.output/'validation_plans.json').read_bytes()),
            catalog_sha256=sha((a.catalog/'tiles.parquet').read_bytes()),puma_used=False,test_used=False,seconds=time.time()-started))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),completed=len(plans),seconds=time.time()-started));print('EXTENDED_VALIDATION_PASSED',len(plans),flush=True)
    except Exception as exc:dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
