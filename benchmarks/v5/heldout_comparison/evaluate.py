"""Score heldout local targets with a separately bound seen-selected checkpoint."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .audit import FOLDS
from .training import METHODS, TrainingData
from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.metrics import score, VERSION
from benchmarks.v5.fair_comparison.protocol import nested
from benchmarks.v5.fair_comparison.public_baselines import summarize
from benchmarks.v5.fair_comparison.train_adapted import forward


def read(path):return json.loads(Path(path).read_text())


def binding(training_tasks, training_run, test_tasks, method, gate=False):
    if method not in METHODS:raise ValueError('undeclared heldout method')
    train = read(training_tasks/'contract.json'); test = read(test_tasks/'contract.json')
    fold = train.get('heldout_fold')
    if fold not in FOLDS or train.get('heldout_excluded_labels') != FOLDS[fold]['exclude']:
        raise ValueError('training fold/exclusion mismatch')
    if (test.get('heldout_fold') != fold or test.get('checkpoint_selection_allowed') is not False or
        test.get('training_protocol_sha256') != digest(training_tasks/'contract.json') or
        Path(test['training_protocol']).resolve() != training_tasks.resolve()):
        raise ValueError('heldout test/training binding mismatch')
    if test.get('heldout_test_targets') != FOLDS[fold]['test'] or test['catalog'] == train['catalog']:
        raise ValueError('original-label terminal test required')
    completed = read(training_run/'summary.json'); run = read(training_run/'run.json')
    if completed['status'] != 'complete' or bool(completed['gate']) != gate:
        raise ValueError('completed training with matching gate scope required')
    if not gate and (completed['step'] != 10000 or not completed['full_manifest_parity']):
        raise ValueError('formal heldout adaptation budget incomplete')
    contract = run['contract']
    if (contract.get('heldout_fold') != fold or contract.get('heldout_excluded_labels') != FOLDS[fold]['exclude'] or
        contract['method'] != method or contract['protocol_sha256'] != digest(training_tasks/'contract.json')):
        raise ValueError('checkpoint training identity mismatch')
    best = read(training_run/'best_checkpoint.json')
    history = completed['history']
    if not history or not all(np.isfinite(h['score']) for h in history):raise ValueError('invalid seen selection history')
    selected = max(history, key=lambda h:(h['score'], h['step']))
    if best['step'] != selected['step'] or best['score'] != selected['score']:
        raise ValueError('checkpoint not selected by seen validation history')
    checkpoint = Path(best['path'])
    if checkpoint.parent.resolve() != training_run.resolve() or digest(checkpoint) != best['checkpoint_sha256']:
        raise ValueError('selected checkpoint path/SHA mismatch')
    return dict(fold=fold, checkpoint=str(checkpoint), checkpoint_sha256=best['checkpoint_sha256'],
        training_protocol_sha256=digest(training_tasks/'contract.json'), test_protocol_sha256=digest(test_tasks/'contract.json'),
        training_contract=contract, selected_seen_step=best['step'], selected_seen_score=best['score'], gate=gate)


def load_selected(training_tasks, training_run, test_tasks, method, device, gate=False):
    identity = binding(training_tasks, training_run, test_tasks, method, gate)
    TrainingData(training_tasks)  # Validate masked seen selection independently of test labels.
    data = LocalData(test_tasks)
    if any(p['target'] not in FOLDS[identity['fold']]['test'][p['dataset']] for p in data.validation):
        raise ValueError('non-heldout target in terminal test')
    saved = torch.load(identity['checkpoint'], map_location='cpu', weights_only=False)
    if saved['contract'] != identity['training_contract'] or saved['step'] != identity['selected_seen_step']:
        raise ValueError('saved checkpoint does not match seen-selected training run')
    model = build(method, device);model.load_state_dict(saved['model'], strict=True);model.eval()
    return model, data, identity


def run(a):
    a.output.mkdir(parents=True, exist_ok=False);torch.set_num_threads(4)
    device = torch.device('cuda:0'); started = time.time()
    model, data, identity = load_selected(a.training_tasks, a.training_run, a.tasks, a.method, device, a.gate)
    plans = read(a.tasks/'validation_maximum.json')
    if a.gate:
        plans = [next(p for p in plans if p['dataset'] == d and p['prompt_group'] == 0) for d in sorted({p['dataset'] for p in plans})]
    write(a.output/'run.json', identity | dict(method=a.method,worker_pid=os.getpid(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        metric_version=VERSION,scope='Terminal original-label heldout LOCAL scoring; no checkpoint/threshold/candidate selection from test GT',
        source_sha256={str(p):digest(p) for p in (Path(__file__), Path(__file__).with_name('training.py'))}))
    rows = []
    with (a.output/'rows.jsonl').open('x') as handle, torch.inference_mode():
        for maximum in plans:
            for budget in data.contract['report_budgets']:
                plan = nested(maximum, budget)
                raw, prompts, supervision = data.materialize(plan, device, model.contexts, model.cells)
                torch.cuda.synchronize(); tick = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):result = forward(model, raw, prompts, plan)
                probability = result['pixel_probability'].float();torch.cuda.synchronize()
                elapsed = 1000*(time.perf_counter()-tick)
                if not torch.isfinite(probability).all():raise ValueError('nonfinite heldout inference')
                record = score(probability[0].cpu().numpy() >= .5, supervision['truth'][0].cpu().numpy(),
                    supervision['valid'][0].cpu().numpy(), supervision['mpp'])
                meta = data.rows[plan['query_id']]
                row = plan | record | dict(method=a.method,latency_ms=elapsed,
                    space_key=meta['space_key'],source_id=meta['source_id'],spatial_group_id=meta['spatial_group_id'])
                rows.append(row);handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            if len(rows)%80 == 0:write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),rows=len(rows),total=4*len(plans)))
    write(a.output/'summary.json',dict(status='complete',worker_pid=os.getpid(),gate=a.gate,method=a.method,
        metrics=summarize(rows),rows=len(rows),rows_sha256=digest(a.output/'rows.jsonl'),seconds=time.time()-started))
    write(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),rows=len(rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('training-tasks','training-run','tasks','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--gate',action='store_true')
    run(p.parse_args())
