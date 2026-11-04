"""Fixed train-only interventions on the accepted final SAM checkpoint, no fitting."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.v5.sam_prior_repair.model import DecoderData, DecoderModel
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.architecture_bridge.train_monitor import probability_diagnostics
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accept', action='store_true')
    args = parser.parse_args()
    settings = json.loads((args.study / 'settings.json').read_text())
    data = DecoderData(settings['cache'], 'sam_native')
    checkpoint = args.study / 'sam_native/checkpoint.pt'
    if args.accept:
        report = json.loads((args.output / 'report.json').read_text())
        assert report['checkpoint_sha256'] == sha(checkpoint)
        assert report['model_state_unchanged'] and report['optimizer_updates'] == 0
        masks = np.load(args.output / 'masks.npz')['masks']
        assert masks.shape == (32, 3, 32768)
        for row, packed in zip(report['rows'], masks):
            gt, mpp = data.labels(row['plan']['query_id'])
            for index, name in enumerate(report['variants']):
                mask = np.unpackbits(packed[index], count=512*512).reshape(512, 512).astype(bool)
                assert score(mask, gt == row['plan']['target'], gt != 255, mpp) == row['metrics'][name]
        receipt = dict(status='accepted_frozen_SAM_train_diagnostic', masks_replayed=96,
                       report_sha256=sha(args.output / 'report.json'), masks_sha256=sha(args.output / 'masks.npz'),
                       checkpoint_sha256=sha(checkpoint), source_sha256=sha(__file__))
        (args.output / 'acceptance.json').write_text(json.dumps(receipt, indent=2))
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(exist_ok=False)
    monitors = {str(n): json.loads((args.study / f'sam_native/train_monitor_{n}.json').read_text())
                for n in [0, 250, 500, 1000]}
    tasks = [r['plan'] for r in monitors['1000']['rows']]
    assert len(tasks) == 32
    assert all([r['plan'] for r in m['rows']] == tasks for m in monitors.values())
    model = DecoderModel('sam_native', settings['seed'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    model.eval()
    before = tree_sha(model.state_dict())
    variants = ['native_decoder', 'combined_prior_direct', 'raw_cosine_direct']
    rows, packed = [], []
    started = time.monotonic()
    with torch.no_grad():
        for index, plan in enumerate(tasks):
            assert data.rows[plan['support_id']]['split'] == data.rows[plan['query_id']]['split'] == 'train'
            inputs = data.model_inputs(plan, 'xcell', 'cuda:0')
            prompts = inputs['prompts']
            se = model.encode(inputs['support'])
            qe = se if inputs['shared'] else model.encode(inputs['query'])
            native = model.decode(se, qe, prompts, inputs['local'])['pixel_logits']
            swapped = dict(prompts)
            for key in ['xy', 'mask']:
                swapped['positive_'+key], swapped['negative_'+key] = prompts['negative_'+key], prompts['positive_'+key]
            other = model.decode(se, qe, swapped, inputs['local'])['pixel_probability']
            source, target = F.normalize(se[1].float(), dim=1), F.normalize(qe[1].float(), dim=1)
            evidence = []
            for sign in ['positive', 'negative']:
                desc = F.normalize(sample_assignment_at_cells(source, prompts[sign+'_xy']), dim=1)
                sim = torch.einsum('bcp,bchw->bphw', desc, target)
                valid_points = prompts[sign+'_mask'].bool()
                evidence.append((sim*valid_points[:, :, None, None]).sum(1)/valid_points.sum(1)[:, None, None])
            raw = F.interpolate((4*(evidence[0]-evidence[1]))[:, None], size=(512, 512), mode='bilinear', align_corners=False)[:, 0]
            zero = native.new_zeros(1, 512, 512)
            learned = model.matcher(se[1], qe[1], prompts, dict(pixel_logits=zero, pixel_probability=zero+.5))['pixel_logits']
            logits = [native, raw+learned, raw]
            assert all(torch.isfinite(v).all() for v in logits)
            # Labels are accessed only after every intervention has been fixed and inferred.
            gt, mpp = data.labels(plan['query_id'])
            truth, valid = torch.from_numpy(gt == plan['target']).cuda(), torch.from_numpy(gt != 255).cuda()
            diagnostic = probability_diagnostics(native[0].sigmoid(), truth, valid, other[0])
            assert diagnostic == monitors['1000']['rows'][index]['diagnostics'], 'Final native prediction monitor must replay exactly'
            masks = [(v[0] >= 0).cpu().numpy() for v in logits]
            rows.append(dict(plan=plan, metrics={name: score(mask, gt == plan['target'], gt != 255, mpp)
                                                for name, mask in zip(variants, masks)}, native_diagnostics=diagnostic))
            packed.append(np.stack([np.packbits(mask) for mask in masks]))
            status = dict(stage='diagnosing', tasks=index+1, total=32, worker_pid=os.getpid(), seconds=time.monotonic()-started)
            (args.output / 'status.json').write_text(json.dumps(status))
            print(json.dumps(status), flush=True)
    assert tree_sha(model.state_dict()) == before
    groups = {}
    for domain in ['crc', 'tiger']:
        for kind in ['local', 'cross_present', 'cross_absent']:
            subset = [r for r in rows if r['plan']['dataset'] == domain and r['plan']['kind'] == kind]
            metric = 'false_positive_fraction' if kind == 'cross_absent' else 'dice'
            groups[domain+'|'+kind] = {v: float(np.mean([r['metrics'][v][metric] for r in subset])) for v in variants}
    np.savez_compressed(args.output / 'masks.npz', masks=np.stack(packed))
    report = dict(status='complete_unaccepted', variants=variants, rows=rows, groups=groups,
                  fixed_train_curves={n: m['groups'] for n, m in monitors.items()}, train_only=True,
                  model_state_unchanged=True, optimizer_updates=0, native_monitor_exact=True,
                  threshold_logit=0, checkpoint_sha256=sha(checkpoint), source_sha256=sha(__file__),
                  monitor_sha256={n: sha(args.study / f'sam_native/train_monitor_{n}.json') for n in monitors},
                  limitation='32 train tasks; direct priors are uncalibrated diagnostics, not optimized segmentation baselines; no causal proof or validation claim')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
