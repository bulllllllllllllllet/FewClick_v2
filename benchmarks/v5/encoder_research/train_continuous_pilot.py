"""Matched three-arm continuation; gate first, fixed endpoint and full dev scoring."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .continuous_data import ContinuousData, sha
from .continuous_prompt import ARMS, ContinuousPromptDecoder, fresh_optimizer, initialize, mask_loss, tree_sha
from .shared_prompt import SharedPromptDecoder
from benchmarks.v5.fair_comparison.metrics import score


def gate_indices(lines):
    plans = [json.loads(line) for line in lines]
    selected = []
    for dataset in ['crc', 'tiger']:
        for category in ['local', 'present', 'absent']:
            candidates = [i for i, p in enumerate(plans) if p['dataset'] == dataset and
                          (p['kind'] == 'local' if category == 'local' else p['kind'].endswith(category))]
            assert candidates
            selected.append(candidates[0])
    covered = {plans[i]['kind'] for i in selected}
    for kind in sorted({p['kind'] for p in plans} - covered):
        selected.append(next(i for i, p in enumerate(plans) if p['kind'] == kind))
    for i in range(len(plans)):
        if len(selected) == 8:
            break
        if i not in selected:
            selected.append(i)
    assert len(selected) == len(set(selected)) == 8
    return selected


def source_identity():
    return {name: sha(Path(__file__).with_name(name)) for name in
            ['train_continuous_pilot.py', 'continuous_prompt.py', 'continuous_data.py', 'shared_prompt.py']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['images', 'grids', 'parent', 'training', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--gate-acceptance', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = ContinuousData(args.images, args.grids)
    parent_report = json.loads((args.parent.parent / 'report.json').read_text())
    parent_receipt = json.loads((args.parent.parent / 'independent_acceptance.json').read_text())
    assert parent_receipt['status'] == 'accepted_fixed_image_pilot'
    assert parent_report['checkpoint_sha256'] == sha(args.parent)
    assert parent_report['cache_index_sha256'] == data.identity['image_index_sha256']
    task_proof = json.loads((args.training.parent / 'independent_acceptance.json').read_text())
    assert task_proof['status'] == 'accepted_train_only_task_factors'
    assert task_proof['files'][args.training.name] == sha(args.training)
    lines = args.training.read_text().splitlines(keepends=True)
    assert len(lines) == 4000
    selected = gate_indices(lines)
    if not args.gate:
        assert args.gate_acceptance is not None
        gate = json.loads(args.gate_acceptance.read_text())
        assert gate['status'] == 'accepted_all_three_continuous_prompt_gates'
        assert gate['sources'] == source_identity()
        assert gate['parent_sha256'] == sha(args.parent) and gate['training_sha256'] == sha(args.training)
        assert gate['inputs'] == data.identity
        assert gate['acceptance_source_sha256'] == sha(Path(__file__).with_name('accept_continuous_pilot.py'))
    model, optimizer, identity = initialize(args.parent, args.arm, args.seed, 'cuda:0')
    baseline = SharedPromptDecoder(1024).to('cuda:0').eval().requires_grad_(False)
    baseline.load_state_dict(torch.load(args.parent, map_location='cuda:0', weights_only=True)['model'], strict=True)
    with torch.inference_mode():
        for index in selected:
            plan = json.loads(lines[index])
            inputs = data.model_inputs(plan, args.arm, 'cuda:0')
            old_inputs = {k: v for k, v in inputs.items() if k not in ['query_cells', 'support_cells']}
            assert torch.equal(model(**inputs), baseline(**old_inputs))
    del baseline
    gc.collect()
    training_lines = [lines[i] for i in selected] if args.gate else lines
    updates = 2 if args.gate else 1000
    status = dict(status='training', arm=args.arm, worker_pid=os.getpid(), updates=0, total_updates=updates)
    def save():
        status['elapsed_seconds'] = time.time() - start
        (args.output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    save()
    losses = []
    with (args.output / 'training_consumed.jsonl').open('w') as stream:
        for step in range(updates):
            optimizer.zero_grad()
            loss_sum = 0.
            for line in training_lines[step * 4:(step + 1) * 4]:
                plan = json.loads(line)
                logits = model(**data.model_inputs(plan, args.arm, 'cuda:0'))
                truth, valid = data.supervision(plan, 'cuda:0')
                loss = mask_loss(logits, truth, valid) / 4
                assert torch.isfinite(loss)
                loss.backward()
                loss_sum += float(loss.detach())
                stream.write(line)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            gradient = float(model.cell_project.weight.grad.abs().max())
            if args.arm == 'image_only':
                assert gradient == 0
            elif args.gate:
                assert gradient > 0
            optimizer.step()
            losses.append(loss_sum)
            if step % 25 == 0 or step + 1 == updates:
                stream.flush()
                status.update(updates=step + 1, loss=loss_sum, cell_gradient_max=gradient)
                save()
                print(json.dumps(status), flush=True)
    cell_changed = bool(torch.count_nonzero(model.cell_project.weight))
    assert cell_changed == (args.arm != 'image_only')
    base_after = {k: v for k, v in model.state_dict().items() if not k.startswith('cell_project.')}
    assert tree_sha(base_after) != identity['parent_model_sha256']
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=2000 + updates,
                    cell_updates=updates, seed=args.seed, arm=args.arm), args.output / 'checkpoint.pt')
    checkpoint = torch.load(args.output / 'checkpoint.pt', map_location='cuda:0', weights_only=True)
    fresh = ContinuousPromptDecoder(args.arm).to('cuda:0').eval()
    fresh.load_state_dict(checkpoint['model'], strict=True)
    restored_optimizer = fresh_optimizer(fresh)
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    assert tree_sha(restored_optimizer.state_dict()) == tree_sha(optimizer.state_dict())
    first = json.loads(training_lines[0])
    sample = data.model_inputs(first, args.arm, 'cuda:0')
    model.eval()
    with torch.no_grad():
        reference = model(**sample)
        assert torch.equal(reference, fresh(**sample))
        perturbed = dict(sample, query_cells=torch.ones_like(sample['query_cells']) * 7,
                         support_cells=torch.ones_like(sample['support_cells']) * -3)
        perturbation = float((fresh(**perturbed) - reference).abs().max())
        assert perturbation == 0 if args.arm == 'image_only' else perturbation > 0
    rows, masks = [], []
    evaluation = [('gate_train', 'recorded', json.loads(line)) for line in training_lines] if args.gate else data.validation()
    status.update(status='scoring')
    save()
    with torch.inference_mode():
        for scope, budget, plan in evaluation:
            prediction = fresh(**data.model_inputs(plan, args.arm, 'cuda:0'))
            assert torch.isfinite(prediction).all()
            mask = (prediction[0] >= 0).cpu().numpy()
            # Scoring GT is loaded only after this prediction; target is never a model input.
            gt, mpp = data.labels(plan['query_id'])
            metrics = score(mask, gt == plan['target'], gt != 255, mpp)
            rows.append(dict(scope=scope, budget=budget, pair_id=plan.get('pair_id'), dataset=plan['dataset'],
                             target=plan['target'], support_id=plan['support_id'], query_id=plan['query_id'],
                             kind=plan['kind'], mask_index=len(masks), **metrics))
            masks.append(np.packbits(mask.reshape(-1)))
            if len(rows) % 50 == 0:
                status['scoring_rows'] = len(rows)
                save()
    np.savez_compressed(args.output / 'packed_masks.npz', masks=np.stack(masks))
    (args.output / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
    continuation = None
    if args.gate:
        truth, valid = data.supervision(first, 'cuda:0')
        for head, opt in [(model, optimizer), (fresh, restored_optimizer)]:
            head.train()
            opt.zero_grad()
            mask_loss(head(**sample), truth, valid).backward()
            opt.step()
        assert tree_sha(model.state_dict()) == tree_sha(fresh.state_dict())
        assert tree_sha(optimizer.state_dict()) == tree_sha(restored_optimizer.state_dict())
        continuation = True
    result = dict(status='complete_continuous_prompt_gate_unaccepted' if args.gate else 'complete_continuous_pilot_unaccepted',
                  arm=args.arm, seed=args.seed, worker_pid=os.getpid(), gate=args.gate,
                  training_updates=updates, consumed_tasks=updates * 4, total_updates=2000 + updates,
                  initial_identity=identity, parent_sha256=sha(args.parent), parent=str(args.parent),
                  training_path=str(args.training), training_sha256=sha(args.training), gate_indices=selected,
                  images=str(args.images), grids=str(args.grids),
                  consumed_sha256=sha(args.output / 'training_consumed.jsonl'), sources=source_identity(), inputs=data.identity,
                  checkpoint_sha256=sha(args.output / 'checkpoint.pt'), cell_parameters_changed=cell_changed,
                  initial_parent_logits_exact=True, exact_disk_reload=True, exact_optimizer_reload=True,
                  exact_optimizer_continuation=continuation, perturbation_max_abs=perturbation, losses=losses,
                  scoring_rows=len(rows), threshold=.5, elapsed_seconds=time.time() - start,
                  gate_acceptance_sha256=sha(args.gate_acceptance) if args.gate_acceptance else None,
                  gate_acceptance_path=str(args.gate_acceptance) if args.gate_acceptance else None,
                  qualification='fixed endpoint development experiment; no WSI/blind/native-SAM superiority claim')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    status.update(status='complete_unaccepted', scoring_rows=len(rows))
    save()
    print(json.dumps({k: v for k, v in result.items() if k != 'losses'}), flush=True)


if __name__ == '__main__':
    main()
