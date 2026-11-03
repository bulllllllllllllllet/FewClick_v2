"""Read-only P3 checkpoint diagnosis. No training, threshold or model selection."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.v5.architecture_bridge.spatial_controls import SpatialData, SpatialModel
from benchmarks.v5.architecture_bridge.train_monitor import probability_diagnostics
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def capacity_probe(assignment, truth, valid, steps):
    """Train-label-only constrained least squares; not a Dice ceiling or inference."""
    a = F.interpolate(assignment, size=truth.shape, mode='bilinear', align_corners=False)[0]
    a = a / a.sum(0, keepdim=True).clamp_min(1e-6)
    x = a[:, valid].T
    y = truth[valid].float()
    gram = x.T @ x / len(y)
    rhs = x.T @ y / len(y)
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-8)
    coef = torch.full_like(rhs, y.mean())
    for _ in range(steps):
        coef = (coef - (gram @ coef - rhs) / lipschitz).clamp(0, 1)
    p = torch.einsum('khw,k->hw', a, coef)
    pred = p >= .5
    tp = (pred & truth & valid).sum()
    dice = float(2 * tp / ((pred & valid).sum() + truth[valid].sum()))
    hard = a.argmax(0)
    # Weighted CUDA bincount is nondeterministic; this diagnostic count is CPU-only.
    hard_valid = hard[valid].cpu().numpy()
    mass = torch.from_numpy(np.bincount(hard_valid, minlength=a.shape[0])).to(a.device).float()
    positives = torch.from_numpy(np.bincount(hard_valid, weights=y.cpu().numpy(), minlength=a.shape[0])).to(a.device).float()
    hard_pred = (positives / mass.clamp_min(1))[hard] >= .5
    hard_tp = (hard_pred & truth & valid).sum()
    return dict(constrained_projection_Dice=dice,
                constrained_mse=float((p[valid] - y).square().mean()),
                majority_hard_region_Dice=float(2 * hard_tp / ((hard_pred & valid).sum() + truth[valid].sum())),
                projected_gradient_inf=float((coef - (coef - (gram @ coef - rhs) / lipschitz).clamp(0, 1)).abs().max()),
                steps=steps, train_label_only=True, deployable=False, strict_Dice_upper_bound=False)


def run(root, mode):
    c = json.loads((root / 'contract.json').read_text())
    assert c['training_allowed'] is False and c['threshold'] == .5
    assert sha(root / 'plans.jsonl') == c['plans_sha256']
    for name, digest in c['sources'].items():
        assert sha(name) == digest, name
    plans = list(map(json.loads, (root / 'plans.jsonl').read_text().splitlines()))
    if mode == 'gate':
        plans = [p for p in plans if p['rank'] == 0]
        assert len(plans) == 12
    else:
        assert json.loads((root / 'gate' / 'acceptance.json').read_text())['status'] == 'accepted_frozen_checkpoint_diagnostic'
        assert len(plans) == 384
    out = root / mode
    out.mkdir(exist_ok=False)
    started = time.monotonic()

    def progress(**kw):
        kw.update(worker_pid=os.getpid(), seconds=time.monotonic() - started)
        dump(out / 'status.json', kw)
        print(json.dumps(kw), flush=True)

    for arm in c['arms']:
        checkpoint = Path(c['formal']) / arm / 'checkpoint.pt'
        assert sha(checkpoint) == c['checkpoints'][arm]
        data = SpatialData(Path(c['cache']), arm)
        model = SpatialModel(arm, device='cuda:0').eval()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['updates'] == 2000 and saved['arm'] == arm
        model.load_state_dict(saved['model'], strict=True)
        del saved
        initial_sha = tree_sha(model.state_dict())
        encodings, region_logits, shapes = [], [], []
        original_encode = model.core.encode_image

        def capture_encode(self, inputs):
            encoded = original_encode(inputs)
            encodings.append(encoded)
            return encoded

        model.core.encode_image = types.MethodType(capture_encode, model.core)
        head_hook = model.core.decoder.delta_head.register_forward_hook(lambda m, i, o: region_logits.append(o))
        backbone_hook = model.core.encoders[0].backbone.register_forward_hook(lambda m, i, o: shapes.append(list(o['out'].shape)))
        records, masks, support_masks = [], [], []
        reference_rows = json.loads((Path(c['formal']) / arm / 'rows.json').read_text())
        reference_index = {(r['plan']['support_id'], r['plan']['query_id'], r['plan']['target']): i
                           for i, r in enumerate(reference_rows) if r['budget'] == '3+2'}
        with np.load(Path(c['formal']) / arm / 'masks.npz') as f:
            reference_masks = f['masks'].copy()
        with torch.no_grad():
            for index, item in enumerate(plans):
                encodings.clear(); region_logits.clear(); shapes.clear()
                plan = item['plan']
                assert data.rows[plan['query_id']]['split'] == data.rows[plan['support_id']]['split'] == item['split']
                inputs = data.model_inputs(plan, 'xcell', 'cuda:0')
                output = model(**inputs)
                p = output['query']['pixel_probability'][0]
                sp = output['support']['pixel_probability'][0]
                assert torch.isfinite(p).all() and torch.isfinite(sp).all()
                qe = encodings[0] if inputs['shared'] else encodings[1]
                assignment = qe['assignment_low']
                a = F.interpolate(assignment, size=p.shape, mode='bilinear', align_corners=False)
                a = a / a.sum(1, keepdim=True).clamp_min(1e-6)
                coefficients = region_logits[0].squeeze(-1)[:, :assignment.shape[1]].sigmoid()
                projected = torch.einsum('bkhw,bk->bhw', a, coefficients).float().clamp(1e-6, 1-1e-6)[0]
                assert torch.equal(projected, p), 'Instrumented projection must equal actual inference'
                other = p
                if item['swap_probe']:
                    prompts = dict(inputs['prompts'])
                    for suffix in ['xy', 'mask']:
                        prompts['positive_' + suffix], prompts['negative_' + suffix] = prompts['negative_' + suffix], prompts['positive_' + suffix]
                    other = model(**dict(inputs, prompts=prompts))['query']['pixel_probability'][0]
                # GT is read only after all ordinary/swap inference passes.
                gt, mpp = data.labels(plan['query_id'])
                sgt, smpp = data.labels(plan['support_id'])
                truth = torch.from_numpy(gt == plan['target']).to(p.device)
                valid = torch.from_numpy(gt != 255).to(p.device)
                mask = (p >= .5).cpu().numpy()
                packed = np.packbits(mask)
                if item['split'] == 'val':
                    key = (plan['support_id'], plan['query_id'], plan['target'])
                    ri = reference_index[key]
                    assert reference_rows[ri]['plan']['positive_xy'] == plan['positive_xy']
                    assert reference_rows[ri]['plan']['negative_xy'] == plan['negative_xy']
                    assert np.array_equal(packed, reference_masks[ri]), (arm, key)
                diagnostics = probability_diagnostics(p, truth, valid, other)
                if not item['swap_probe']:
                    diagnostics['swap_valid_mean'] = None
                mass = assignment[0].mean((1, 2))
                distribution = dict(assignment_shape=list(assignment.shape), backbone_shapes=shapes[:3],
                                    graph_nodes_per_scale=[x.shape[1] for x in qe['tokens']],
                                    argmax_used_slots=int(torch.unique(assignment.argmax(1)).numel()),
                                    average_assignment_max=float(assignment.max(1).values.mean()),
                                    mass_effective_slots=float(torch.exp(-(mass * mass.clamp_min(1e-9).log()).sum())),
                                    mean_pixel_entropy=float(-(assignment * assignment.clamp_min(1e-9).log()).sum(1).mean()),
                                    region_probability_std=float(coefficients.std(unbiased=False)),
                                    dense_enabled=bool(model.core.use_dense), exact_projection=True)
                record = dict(item, metrics=score(mask, gt == plan['target'], gt != 255, mpp),
                              support_metrics=score((sp >= .5).cpu().numpy(), sgt == plan['target'], sgt != 255, smpp),
                              diagnostics=diagnostics, representation=distribution)
                if item['capacity_probe']:
                    assert item['split'] == 'train'
                    record['capacity'] = capacity_probe(assignment, truth, valid, c['oracle_fit_steps'])
                records.append(record)
                masks.append(packed)
                support_masks.append(np.packbits((sp >= .5).cpu().numpy()))
                if index % 8 == 0 or index + 1 == len(plans):
                    progress(stage='inference', arm=arm, done=index+1, total=len(plans))
        head_hook.remove(); backbone_hook.remove()
        model.core.encode_image = original_encode
        assert tree_sha(model.state_dict()) == initial_sha
        assert sha(checkpoint) == c['checkpoints'][arm]
        dump(out / (arm + '_rows.json'), records)
        np.savez_compressed(out / (arm + '_masks.npz'), masks=np.stack(masks), support_masks=np.stack(support_masks))
        dump(out / (arm + '_receipt.json'), dict(arm=arm, rows=len(records), model_state_unchanged=True,
                                               model_state_sha256=initial_sha, checkpoint_sha256=c['checkpoints'][arm],
                                               rows_sha256=sha(out / (arm + '_rows.json')),
                                               masks_sha256=sha(out / (arm + '_masks.npz'))))
        del model, data, encodings, region_logits, original_encode, projected, a, p, sp, qe, assignment, output, other, inputs, reference_masks
        gc.collect(); torch.cuda.empty_cache()
    progress(stage='complete_unaccepted')


def accept(root, mode):
    c = json.loads((root / 'contract.json').read_text()); out = root / mode
    counts = {}
    for arm in c['arms']:
        data = SpatialData(Path(c['cache']), arm)
        receipt = json.loads((out / (arm + '_receipt.json')).read_text())
        assert receipt['model_state_unchanged']
        assert receipt['checkpoint_sha256'] == sha(Path(c['formal']) / arm / 'checkpoint.pt')
        assert receipt['rows_sha256'] == sha(out / (arm + '_rows.json'))
        assert receipt['masks_sha256'] == sha(out / (arm + '_masks.npz'))
        rows = json.loads((out / (arm + '_rows.json')).read_text())
        with np.load(out / (arm + '_masks.npz')) as f:
            masks, support = f['masks'].copy(), f['support_masks'].copy()
        assert len(rows) == len(masks) == len(support) == (12 if mode == 'gate' else 384)
        for r, packed, spacked in zip(rows, masks, support):
            for field, identity, values in [('metrics', 'query_id', packed), ('support_metrics', 'support_id', spacked)]:
                gt, mpp = data.labels(r['plan'][identity])
                mask = np.unpackbits(values, count=512*512).reshape(512, 512).astype(bool)
                assert score(mask, gt == r['plan']['target'], gt != 255, mpp) == r[field]
            if 'capacity' in r:
                assert r['split'] == 'train' and not r['capacity']['deployable']
        counts[arm] = len(rows)
    dump(out / 'acceptance.json', dict(status='accepted_frozen_checkpoint_diagnostic', mode=mode, rows_per_arm=counts,
                                     masks_independently_replayed=sum(counts.values())*2,
                                     source_sha256=c['sources'], contract_sha256=sha(root/'contract.json'),
                                     training_performed=False, threshold_selected=False, P3_complete=False))
    print(json.dumps(dict(status='accepted_frozen_checkpoint_diagnostic', mode=mode, rows=counts)), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=['gate', 'full', 'accept_gate', 'accept_full'], required=True)
    args = p.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.mode.startswith('accept_'):
        accept(args.root, args.mode.removeprefix('accept_'))
    else:
        run(args.root, args.mode)


if __name__ == '__main__':
    main()
