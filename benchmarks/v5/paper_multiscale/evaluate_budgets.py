"""Same tasks and nested positive prefixes for 1+1, 3+1 and 5+1 prompts."""
import argparse
import json
from pathlib import Path

import pyvips
import torch

from .data import MultiscaleCorpus
from .model import load_model
from .train import ARMS
from benchmarks.v5.multidomain_training.click_policy import repoint
from benchmarks.v5.multidomain_training.evaluate import evaluate
from benchmarks.v5.multidomain_data.prepare import dump,sha


def budget_plans(corpus,plans):
    groups={n:[] for n in (1,3,5)}
    for plan in plans:
        maximum=repoint(corpus,plan|dict(positive_xy=[[0.,0.]]*5,negative_xy=[[0.,0.]]))
        for n in groups:
            groups[n].append(maximum|dict(positive_xy=maximum['positive_xy'][:n],
                                          click_budget=f'{n}+1',click_audit_scope='audit refers to full five-positive sequence; actual prefix stored'))
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);device=torch.device('cuda:0')
    corpus=MultiscaleCorpus(a.catalog);plans=json.loads((a.run/'validation_plans.json').read_text())
    groups=budget_plans(corpus,plans);selected=json.loads((a.run/'best_checkpoint.json').read_text())
    checkpoint=Path(selected['path']);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['contract']['arm']!=a.arm or saved['contract']['catalog_sha256']!=sha((a.catalog/'tiles.parquet').read_bytes()):
        raise ValueError('evaluation model/data identity mismatch')
    model,_=load_model(device,*ARMS[a.arm]);model.load_state_dict(saved['model'],strict=True);del saved
    results={}
    for n,locked in groups.items():
        dump(a.output/f'plans_{n}p1n_1025.json',locked)
        results[str(n)]=evaluate(model,corpus,locked,device,a.output/f'budget_{n}p1n_1025',amp=True)
    dump(a.output/'summary.json',dict(status='complete',arm=a.arm,checkpoint=selected,results=results,
         scope='same development task identities; nested positive prefixes; no test claims'))


if __name__=='__main__':main()
