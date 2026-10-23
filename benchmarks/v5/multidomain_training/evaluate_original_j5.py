"""Evaluate the original retained J5 weights and region projection, without dense."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import torch

from benchmarks.v5.multidomain_data.prepare import dump, sha
from .episodes import EpisodeCorpus
from .evaluate import evaluate
from .model import load_model


class OriginalRegionProjection(torch.nn.Module):
    """Expose the original decoder's projected probability without a residual."""
    def forward(self, features):
        return {'pixel_probability': features['base']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'plans', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if Path('/nfs-medical3/zyh/fewclick_v2') not in a.output.resolve().parents:
        raise ValueError('output outside artifact root')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time(); torch.set_num_threads(4)
    try:
        plans = json.loads(a.plans.read_text())
        if a.smoke:
            selected = []; seen = set()
            for plan in plans:
                key = plan['dataset'], plan['kind']
                if key not in seen:
                    seen.add(key); selected.append(plan)
            plans = selected
        if not plans or any(p['dataset'] not in ('crc','tiger','ignite') or
                            p['support_quality']!='original' or p['query_quality']!='original' for p in plans):
            raise ValueError('expected retained-source original-quality primary tasks')
        device = torch.device('cuda:0')
        corpus = EpisodeCorpus(a.catalog, prompt_policy='representative')
        model, weights = load_model(device, mapper='pixel')
        # load_model provides the retained original J5, never a trained C3.
        model.dense = OriginalRegionProjection().to(device)
        model.eval().requires_grad_(False)
        params = sum(p.numel() for p in model.parameters())
        if params != sum(p.numel() for p in model.j5.parameters()):
            raise ValueError('dense parameters remain in the evaluated model')
        dump(a.output/'run.json', dict(worker_pid=os.getpid(), source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
             source_sha256=sha(Path(__file__).read_bytes()), original_j5_checkpoint=weights['j5_checkpoint'],
             original_j5_sha256=weights['j5_sha256'], catalog_sha256=sha((a.catalog/'tiles.parquet').read_bytes()),
             source_plans_sha256=sha(a.plans.read_bytes()), weights='original retained J5 only',
             output_head='region projection; no dense residual', mapper='pixel', active_parameters=params,
             smoke=a.smoke, amp=True, threshold=.5, training_performed=False))
        dump(a.output/'plans.json', plans)
        dump(a.output/'status.json', dict(status='evaluating', worker_pid=os.getpid(), tasks=len(plans)))
        report = evaluate(model, corpus, plans, device, a.output/'evaluation', amp=True)
        dump(a.output/'summary.json', report)
        dump(a.output/'status.json', dict(status='complete', worker_pid=os.getpid(), tasks=len(plans), seconds=time.time()-started))
        print('ORIGINAL_J5_COMPLETE', json.dumps(report['overall']), flush=True)
    except Exception as exc:
        dump(a.output/'status.json', dict(status='failed', worker_pid=os.getpid(), error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
