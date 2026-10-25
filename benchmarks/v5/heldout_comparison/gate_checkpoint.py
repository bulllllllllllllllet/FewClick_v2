"""Real CPU updates and fresh restore across seen-training/heldout-test contracts."""
import argparse
import gc
import json
import os
from pathlib import Path
import random

import pyvips
import numpy as np
import torch

from .training import TrainingData, fingerprint
from .evaluate import binding, load_selected
from benchmarks.v5.evidence_closure.audit_external import require
from benchmarks.v5.fair_comparison.adapted_models import build
from benchmarks.v5.fair_comparison.audit import digest, write
from benchmarks.v5.fair_comparison.local_data import LocalData
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.fair_comparison.train_adapted import forward, loss_for
from benchmarks.v5.multidomain_training.distributed_utils import optimizer_digest


def save_cpu(path,model,optimizer,step,contract,history):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,contract=contract,history=history,
        rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())),path)


def restore_cpu(path,model,optimizer,contract):
    saved = torch.load(path,map_location='cpu',weights_only=False)
    require(saved['contract'] == contract,'CPU checkpoint identity')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng']['torch']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python'])
    return saved['step']


def run(a):
    require(not torch.cuda.is_available(), 'CPU gate must hide CUDA')
    a.output.mkdir(parents=True, exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261026)
    device = torch.device('cpu');method = 'fewclick_m3d_image'
    data = TrainingData(a.tasks, a.realized)
    training = [p for i in range(2) for d in ('crc','tiger') for p in [
        [v for v in data.training if v['dataset'] == d][i]]]
    seen = [next(p for p in data.validation if p['dataset'] == d) for d in ('crc','tiger')]
    model = build(method, device);optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=1e-4)
    contract = fingerprint(model,data,method,2,1,2,20261026)
    write(a.output/'run.json',dict(contract=contract,worker_pid=os.getpid(),scope='CPU two-update contract gate, not formal accuracy or full native GPU trainer'))
    before = [e.backbone.layer3[0].conv1.weight.detach().clone() for e in model.core.encoders]
    history = [];updates = []
    for step in range(2):
        model.set_stage(step == 1);model.train();optimizer.zero_grad(set_to_none=True)
        for plan in training[2*step:2*step+2]:
            raw,prompts,supervision = data.materialize(plan,device,model.contexts,model.cells)
            require(not any(bool((supervision['labels'] == x).any()) for x in data.contract['heldout_excluded_labels'][plan['dataset']]), 'excluded supervision')
            result = forward(model,raw,prompts,plan);loss = loss_for(model,result,supervision,prompts)
            require(torch.isfinite(loss),'finite masked objective');(loss/2).backward()
        norms = {g['name']:sum(float(p.grad.detach().float().norm()) for p in g['params'] if p.grad is not None) for g in optimizer.param_groups}
        require(norms['heads'] > 0 and np.isfinite(list(norms.values())).all(),'finite head gradients')
        require(norms['backbone'] > 0 if step == 1 else norms['backbone'] == 0,'staged backbone gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates.append(dict(step=step+1,norms=norms))
        model.eval();scores = []
        with torch.inference_mode():
            for plan in seen:
                raw,prompts,sup = data.materialize(plan,device,model.contexts,model.cells)
                probability = forward(model,raw,prompts,plan)['pixel_probability']
                record = score(probability[0].numpy() >= .5,sup['truth'][0].numpy(),sup['valid'][0].numpy(),sup['mpp'])
                scores.append(record['dice'])
        value = float(np.mean(scores));history.append(dict(step=step+1,score=value))
        checkpoint = a.output/f'checkpoint_step_{step+1:05d}.pth'
        save_cpu(checkpoint,model,optimizer,step+1,contract,history)
        if value >= max(h['score'] for h in history):
            write(a.output/'best_checkpoint.json',dict(path=str(checkpoint),step=step+1,score=value,checkpoint_sha256=digest(checkpoint)))
        print(json.dumps(dict(stage='updated_and_seen_selected',step=step+1,norms=norms)),flush=True)
    require(all(not torch.equal(old,e.backbone.layer3[0].conv1.weight) for old,e in zip(before,model.core.encoders)), 'three view backbone updates')
    write(a.output/'summary.json',dict(status='complete',gate=True,step=2,history=history,full_manifest_parity=False,
        scope='Two-update CPU contract gate only; full native trainer/accuracy pending'))
    identity = binding(a.tasks,a.output,a.test_tasks,method,gate=True)
    test_data = LocalData(a.test_tasks)
    tests = [next(p for p in test_data.validation if p['dataset'] == d) for d in sorted({p['dataset'] for p in test_data.validation})]
    saved = torch.load(identity['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    expected_optimizer = optimizer_digest(optimizer.state_dict());predictions = {}
    model.eval()
    with torch.inference_mode():
        for plan in tests:
            raw,prompts,sup = test_data.materialize(plan,device,model.contexts,model.cells)
            require(bool(sup['truth'].any()),'original heldout positive GT available for score')
            predictions[plan['pair_id']] = forward(model,raw,prompts,plan)['pixel_probability'].clone()
    del model,optimizer,result,loss,saved,before;gc.collect()
    model,test_data,loaded = load_selected(a.tasks,a.output,a.test_tasks,method,device,gate=True)
    optimizer = torch.optim.AdamW(model.parameter_groups(),weight_decay=1e-4)
    restored_step = restore_cpu(Path(identity['checkpoint']),model,optimizer,contract)
    require(optimizer_digest(optimizer.state_dict()) == expected_optimizer,'fresh optimizer restore')
    require(restored_step == identity['selected_seen_step'],'fresh restore step')
    model.eval();replays = []
    with torch.inference_mode():
        for plan in tests:
            raw,prompts,sup = test_data.materialize(plan,device,model.contexts,model.cells)
            probability = forward(model,raw,prompts,plan)['pixel_probability']
            require(torch.equal(probability,predictions[plan['pair_id']]),'fresh heldout prediction exact replay')
            path = a.output/f'gate_probability_{plan["pair_id"]}.npy';np.save(path,probability[0].numpy())
            require(np.array_equal(np.load(path,allow_pickle=False),probability[0].numpy()),'saved probability replay')
            replays.append(dict(dataset=plan['dataset'],pair_id=plan['pair_id'],probability=str(path),
                sha256=digest(path),original_GT_positive_pixels=int(sup['truth'].sum()),fresh_prediction_max_error=0.))
    rejected = []
    invalid = a.output/'invalid_test';invalid.mkdir()
    bad = json.loads((a.test_tasks/'contract.json').read_text());bad['checkpoint_selection_allowed'] = True
    write(invalid/'contract.json',bad)
    for name,test,gate in [('test_selection_forbidden',invalid,True),('gate_checkpoint_not_formal',a.test_tasks,False)]:
        try:binding(a.tasks,a.output,test,method,gate)
        except ValueError:rejected.append(name)
        else:raise ValueError('invalid binding accepted: '+name)
    require(len(rejected) == 2,'negative binding checks')
    write(a.output/'gate_acceptance.json',dict(status='passed_checkpoint_binding_gate',worker_pid=os.getpid(),updates=updates,
        replays=replays,rejected=rejected,selected_seen_step=loaded['selected_seen_step'],checkpoint_sha256=loaded['checkpoint_sha256'],
        source_sha256={str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('training.py'),Path(__file__).with_name('evaluate.py'))},
        scope='Actual masked head/joint CPU updates, fresh model/optimizer/RNG restore, originalGT heldout scorer; not formal model accuracy'))
    write(a.output/'status.json',dict(status='passed_checkpoint_binding_gate',worker_pid=os.getpid()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('tasks','realized','test-tasks','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
