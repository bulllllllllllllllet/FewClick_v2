"""Frozen correct/wrong-target/zero-task controls on existing remote pairs."""
import argparse
from collections import Counter, OrderedDict, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips  # Load before torchvision.
import numpy as np
import torch
from . import module_source
from .boundary_data import load_baseline, make_loader
from .continue_dense_boundary import DenseContextModel
from .evaluate_dense_transfer import ROOT, DENSE_RUN, PROMPT_KEYS, sha256
from .probe_boundary_refinement import metrics, summarize
from .support_query_dense import encode_support, predict_query

SOURCE = ROOT/'support_query/dense_transfer_1016_115814/formal_1016_115814'


def reconstruct_index(pairs, source):
    """Original writer fetched query then support, numbering first patch use."""
    result = {}
    for pair in pairs:
        for role in ('query', 'support'):
            pid = pair[role+'_id']
            if pid not in result:
                result[pid] = str(source/'encoded'/f'{len(result):05d}.pth')
    if set(map(Path, result.values())) != set((source/'encoded').glob('*.pth')):
        raise ValueError('reconstructed feature index does not cover cache exactly')
    return result


def select_wrong(episodes, pairs):
    """Same support image, size and positive budget; different source class."""
    candidates = defaultdict(list)
    for i in range(len(episodes)):
        slots = episodes.positive_slot_sets[i]
        n = len(json.loads(slots) if isinstance(slots, str) else slots)
        key = (str(episodes.patch_ids[i]), str(episodes.prompt_sizes[i]), n)
        candidates[key].append(i)
    selected, skipped = [], []
    for pair in pairs:
        key = (pair['support_id'], pair['prompt_size'], pair['positive_count'])
        options = [i for i in candidates[key] if int(episodes.target_classes[i]) != pair['target_class']]
        # One deterministic alternative, no choice by predictions or query GT.
        options.sort(key=lambda i: (int(episodes.target_classes[i]), i))
        if options:
            i = options[(pair['position'] + pair['target_class']) % len(options)]
            selected.append(pair | {'wrong_index': i, 'wrong_target_class': int(episodes.target_classes[i])})
        else:
            selected.append(pair | {'wrong_index': None, 'wrong_target_class': None})
            skipped.append({'source_number': pair['source_number'], 'reason': 'no different-class episode with identical source/size/count'})
    return selected, skipped


def zero_task(task):
    """Explicit out-of-distribution ablation, NOT an empty-prompt encoder call."""
    return {k: torch.zeros_like(v) for k, v in task.items()}


def controls_report(rows):
    report = {}
    for protocol in sorted({r['protocol'] for r in rows}):
        report[protocol] = {}
        for model in ('j5', 'dense'):
            rr = [r for r in rows if r['protocol'] == protocol and r['model'] == model]
            correct = {r['source_number']: r for r in rr if r['condition'] == 'correct'}
            output = {}
            for condition in ('correct', 'wrong', 'zero_task'):
                cc = [r for r in rr if r['condition'] == condition]
                if not cc:
                    output[condition] = {'episodes': 0}; continue
                stats = summarize([r | {'prompt_violations': 0} for r in cc]); stats.pop('prompt_violations')
                stats['per_class'] = {str(c): {'episodes': len(cr), 'dice': float(np.mean([r['dice'] for r in cr]))}
                    for c in sorted({r['target_class'] for r in cc})
                    for cr in [[r for r in cc if r['target_class'] == c]]}
                if condition != 'correct':
                    stats['correct_dice_on_same_subset'] = float(np.mean([correct[r['source_number']]['dice'] for r in cc]))
                    for metric in ('dice', 'bf1_2'):
                        evaluable = [r for r in cc if r[metric] is not None and correct[r['source_number']][metric] is not None]
                        if not evaluable:
                            stats[metric+'_delta'] = {'evaluable_pairs': 0}; continue
                        grouped = defaultdict(list)
                        for r in evaluable:
                            grouped[r['query_spatial_group']].append(r[metric]-correct[r['source_number']][metric])
                        values = [grouped[g] for g in sorted(grouped)]
                        sums = np.array([sum(v) for v in values]); counts = np.array([len(v) for v in values])
                        rng = np.random.default_rng(20261016)
                        draws = rng.integers(0, len(values), (2000, len(values)))
                        boot = sums[draws].sum(1)/counts[draws].sum(1)
                        stats[metric+'_delta'] = {'control_minus_correct': float(sums.sum()/counts.sum()),
                            'evaluable_pairs': len(evaluable), 'spatial_blocks': len(values),
                            'spatial_block_bootstrap_95': np.percentile(boot, [2.5, 97.5]).tolist()}
                    for key in ('changed_fraction', 'mask_iou_with_correct', 'probability_mae'):
                        stats[key] = float(np.mean([r[key] for r in cc]))
                output[condition] = stats
            report[protocol][model] = output
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if ROOT not in args.output.resolve().parents:
        raise ValueError('artifact root')
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'masks').mkdir()
    started = time.time(); torch.set_num_threads(4); device = torch.device('cuda:0')
    original_run = json.loads((SOURCE/'run.json').read_text())
    completed = [json.loads(x) for x in (SOURCE/'completed.jsonl').read_text().splitlines()]
    if len(completed) != len(original_run['pairs']):
        raise ValueError('incomplete source run')
    for i, (pair, done) in enumerate(zip(original_run['pairs'], completed)):
        if done['number'] != i or any(done[k] != v for k, v in pair.items()):
            raise ValueError('source completion identity mismatch')
    index = reconstruct_index(original_run['pairs'], SOURCE)
    source = json.loads((DENSE_RUN/'run.json').read_text())
    original = json.loads((Path(source['source_cache'])/'run.json').read_text())
    metadata = json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
    data_args = argparse.Namespace(**original['args'])
    data_args.spatial_data = Path(data_args.spatial_data); data_args.routing = Path(data_args.routing)
    episodes = make_loader(data_args, metadata, 'val').dataset.dataset.episodes
    pairs = [p | {'source_number': i} for i, p in enumerate(completed)
             if p['protocol'] in ('same_slide_present', 'cross_slide_present')]
    pairs, skipped = select_wrong(episodes, pairs)
    if args.smoke:
        chosen = {}
        for p in pairs:
            if p['wrong_index'] is not None:
                chosen.setdefault((p['protocol'], p['target_class']), p)
        pairs = list(chosen.values())
        if len({p['protocol'] for p in pairs}) != 2 or len({p['target_class'] for p in pairs}) < 2:
            raise ValueError('smoke coverage')
    manifest = {'source_run': str(SOURCE), 'source_run_sha256': sha256(SOURCE/'run.json'),
        'source_revision': subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        'source_sha256': sha256(module_source('evaluate_prompt_dependence')),
        'checkpoint': original_run['dense_checkpoint'], 'checkpoint_sha256': sha256(Path(original_run['dense_checkpoint'])),
        'pairs': pairs, 'skipped_wrong': skipped, 'reconstructed_feature_index': index,
        'smoke': args.smoke, 'train': False, 'test_used': False, 'threshold': .5,
        'zero_task_policy': 'all q_positive/q_negative/task_token zero; native empty support unsupported',
        'wrong_policy': 'same source patch, different class, same size and positive/negative count',
        'scope': original_run['scope'], 'worker_pid': os.getpid()}
    (args.output/'run.json').write_text(json.dumps(manifest, indent=2))
    j5 = load_baseline(metadata, Path(source['checkpoint']), device).eval().requires_grad_(False)
    dense = DenseContextModel().to(device).eval().requires_grad_(False)
    dense.load_state_dict(torch.load(Path(original_run['dense_checkpoint']), map_location=device, weights_only=False)['model'], strict=True)
    cache = OrderedDict(); rows = []
    def encoded(pid):
        if pid not in cache:
            cache[pid] = torch.load(index[pid], map_location=device, weights_only=False)
            if len(cache) > 8: cache.popitem(last=False)
        cache.move_to_end(pid); return cache[pid]
    def to_device(prompts):
        return {k: v.to(device) for k, v in prompts.items()}
    for number, pair in enumerate(pairs):
        qe, se = encoded(pair['query_id']), encoded(pair['support_id'])
        prompts = to_device({k: torch.tensor(pair['support_prompts'][k]) for k in PROMPT_KEYS})
        # torch.tensor preserves bool mask lists and integer prompt_size_id.
        tasks = {'correct': encode_support(j5, se, prompts)}
        tasks['zero_task'] = zero_task(tasks['correct'])
        if pair['wrong_index'] is not None:
            episode = episodes[pair['wrong_index']]
            if str(episode['patch_id']) != pair['support_id']:
                raise ValueError('wrong support identity')
            wp = to_device({k: episode[k].unsqueeze(0) for k in PROMPT_KEYS})
            if any(int(wp[s+'_mask'].sum()) != int(prompts[s+'_mask'].sum()) for s in ('positive','negative')):
                raise ValueError('wrong annotation budget mismatch')
            if not torch.equal(wp['prompt_size_id'], prompts['prompt_size_id']):
                raise ValueError('wrong prompt size mismatch')
            tasks['wrong'] = encode_support(j5, se, wp)
            pair['wrong_prompts'] = {k: v.cpu().tolist() for k, v in wp.items()}
        with np.load(SOURCE/'masks'/f"{pair['source_number']:05d}.npz") as old:
            truth = torch.from_numpy(old['truth'].copy()).to(device)[None]
            valid = torch.from_numpy(old['valid'].copy()).to(device)[None]
            original_masks = {m: old[m].copy() for m in ('j5','dense')}
        arrays = {'truth': truth.cpu().numpy()[0], 'valid': valid.cpu().numpy()[0]}
        probabilities = {}
        with torch.inference_mode():
            for condition, task in tasks.items():
                prediction, features = predict_query(j5, dense, qe, task, prompts,
                    query_id=pair['query_id'], support_id=pair['support_id'])
                if features['prompt_maps'].count_nonzero(): raise ValueError('remote heatmap')
                probabilities[condition] = prediction
                for model, probability in prediction.items():
                    if not torch.isfinite(probability).all(): raise FloatingPointError('prediction')
                    mask = (probability >= .5).cpu().numpy()[0]
                    if condition == 'correct' and not np.array_equal(mask, original_masks[model]):
                        raise ValueError('correct support failed exact binary-mask replay')
                    arrays[condition+'_'+model] = mask
                    row = {k: pair[k] for k in ('source_number','position','protocol','query_id','support_id','target_class','query_spatial_group','wrong_target_class')}
                    row.update(condition=condition, model=model, mask_file=f'masks/{number:05d}.npz', **metrics(probability, truth, valid))
                    if condition != 'correct':
                        cp = probabilities['correct'][model]; a = (probability >= .5) & valid; b = (cp >= .5) & valid
                        union = int((a | b).sum())
                        row.update(changed_fraction=float((a != b).sum()/valid.sum().clamp_min(1)),
                            mask_iou_with_correct=float((a & b).sum())/union if union else 1.,
                            probability_mae=float((probability-cp).abs()[valid].mean()))
                    rows.append(row)
        path = args.output/'masks'/f'{number:05d}.npz'
        np.savez_compressed(path, **arrays)
        # Reopen output and verify all reported pixel counts before completion.
        with np.load(path) as saved:
            v, t = saved['valid'], saved['truth'] & saved['valid']
            current = rows[-2*len(tasks):]
            for r in current:
                p = saved[r['condition']+'_'+r['model']] & v
                counts = (int((p & t).sum()), int((p & ~t & v).sum()), int((~p & t).sum()))
                if counts != (r['tp'],r['fp'],r['fn']): raise ValueError('saved mask counts')
        with (args.output/'metrics.jsonl').open('a') as f:
            for r in current: f.write(json.dumps(r)+'\n')
        with (args.output/'completed.jsonl').open('a') as f: f.write(json.dumps(pair)+'\n')
        status = {'status':'evaluating','worker_pid':os.getpid(),'completed':number+1,'pairs':len(pairs),'elapsed_seconds':time.time()-started}
        (args.output/'status.json').write_text(json.dumps(status))
        print('PAIR',number+1,'/',len(pairs),pair['protocol'],'wrong',pair['wrong_target_class'],'seconds',round(time.time()-started,1),flush=True)
    report = {'summary':controls_report(rows), 'pairs':len(pairs),
        'pair_counts':dict(Counter(p['protocol'] for p in pairs)),
        'wrong_available':sum(p['wrong_index'] is not None for p in pairs),
        'correct_binary_replay_exact':True,'saved_mask_metrics_verified':True,
        'elapsed_seconds':time.time()-started,'scope':manifest['scope'],'smoke':args.smoke}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid(),'completed':len(pairs)}))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
