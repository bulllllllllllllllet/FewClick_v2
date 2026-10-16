"""Frozen dense local/remote/absent/cross-WSI diagnostics; never train or read test."""
import argparse
from collections import OrderedDict, defaultdict
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time

import pyvips  # Load libvips/libjpeg before torchvision
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from . import module_source
from .boundary_data import load_baseline, make_loader, move_batch
from .continue_dense_boundary import DenseContextModel, dense_loss
from .probe_boundary_refinement import metrics, summarize
from .support_query_dense import encode_image, encode_support, predict_query
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import collate_joint_pixel_episodes
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import remap_prompt_tokens
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts

ROOT = Path('/nfs-medical3/zyh/fewclick_v2')
DENSE_RUN = ROOT/'boundary_refinement/next_stage_1014_200033/matched96k_1014_221710'
PROMPT_KEYS = ('positive_xy', 'negative_xy', 'positive_mask', 'negative_mask', 'prompt_size_id')
IMAGE_KEYS = ('image', 'cells', 'cell_valid', 'total_cell_count', 'fine_active')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def build_pairs(episodes, refs, spatial, patches, seed=20261016):
    """GT/ontology used ONLY to define simulated evaluation tasks, before inference."""
    fields = ['context_x0', 'context_y0', 'context_x1', 'context_y1']
    box = lambda row: [float(row[k]) for k in fields]
    def count(slots):
        return len(json.loads(slots) if isinstance(slots, str) else slots)
    keys = [(str(episodes.wsi_ids[i]), int(episodes.target_classes[i]),
             str(episodes.prompt_sizes[i]), count(episodes.positive_slot_sets[i])) for i in range(len(episodes))]
    same, other = defaultdict(list), defaultdict(list)
    unique = {}
    for i, key in enumerate(keys):
        same[key].append(i); other[key[1:]].append(i)
        unique.setdefault(str(episodes.patch_ids[i]), i)
    by_wsi = defaultdict(list)
    for pid, i in unique.items():
        by_wsi[str(episodes.wsi_ids[i])].append(i)
    pairs, skipped = [], []
    rng = np.random.default_rng(seed)
    for position, ref in enumerate(refs):
        identity = ref['identity']; qi = int(identity['episode_index']); pid = str(episodes.patch_ids[qi])
        if pid != identity['patch_id'] or int(episodes.target_classes[qi]) != identity['target_class']:
            raise ValueError('source episode identity changed')
        qr = spatial.loc[pid]; key = keys[qi]
        if qr.split != 'val':
            raise ValueError('query outside val')
        base = {'position': position, 'target_class': key[1], 'original_query_index': qi,
                'original_query_id': pid, 'prompt_size': key[2], 'positive_count': key[3]}
        pairs.append(base | {'protocol': 'local', 'query_index': qi, 'support_index': qi})
        for protocol, candidates in [('same_slide_present', same[key]),
                                      ('cross_slide_present', other[key[1:]]),
                                      ('same_slide_absent', by_wsi[key[0]])]:
            found = None
            for index in rng.permutation(candidates):
                index = int(index); candidate_pid = str(episodes.patch_ids[index]); cr = spatial.loc[candidate_pid]
                if cr.split != 'val':
                    raise ValueError('support outside val')
                if candidate_pid == pid:
                    continue
                if protocol == 'cross_slide_present':
                    if str(episodes.wsi_ids[index]) == key[0] or cr.source_key == qr.source_key:
                        continue
                elif cr.source_key != qr.source_key or not separated_contexts(box(qr), box(cr), gap=512):
                    continue
                if protocol == 'same_slide_absent':
                    if key[1] in json.loads(patches.loc[candidate_pid, 'present_classes']):
                        continue
                    q, s = index, qi
                else:
                    q, s = qi, index
                found = base | {'protocol': protocol, 'query_index': q, 'support_index': s}
                break
            if found is None:
                skipped.append(base | {'protocol': protocol, 'reason': 'no eligible candidate; constraints retained'})
            else:
                pairs.append(found)
    for pair in pairs:
        for role in ('query', 'support'):
            pid = str(episodes.patch_ids[pair[role+'_index']]); row = spatial.loc[pid]
            pair[role+'_id'] = pid; pair[role+'_wsi'] = str(row.wsi_id)
            pair[role+'_context'] = box(row); pair[role+'_spatial_group'] = str(row.spatial_group_id)
        pair['requested_target_is_model_input'] = False
    return pairs, skipped


def present_summary(rows):
    # Remote query tiles have no local clicks; do not manufacture a violation count.
    result = summarize([r | {'prompt_violations': 0} for r in rows])
    result.pop('prompt_violations')
    for key in ('unprompted_dice', 'unprompted_bf1_2'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key+'_evaluable'] = len(values)
    return result


def verify_saved_masks(path, rows):
    with np.load(path) as saved:
        valid, truth = saved['valid'], saved['truth'] & saved['valid']
        for row in rows:
            pred = saved[row['model']] & valid
            if row['protocol'].endswith('absent'):
                area = float(pred.sum()/max(int(valid.sum()), 1))
                if abs(area-row['false_positive_area']) > 1e-7:
                    raise RuntimeError('saved absent mask metric mismatch')
            else:
                counts = {'tp': int((pred & truth).sum()), 'fp': int((pred & ~truth & valid).sum()),
                          'fn': int((~pred & truth).sum())}
                if any(row[k] != value for k, value in counts.items()):
                    raise RuntimeError('saved present mask metric mismatch')


def summarize_rows(rows):
    output = {}
    for protocol in sorted({r['protocol'] for r in rows}):
        output[protocol] = {}
        for model in ('j5', 'dense'):
            rr = [r for r in rows if r['protocol'] == protocol and r['model'] == model]
            if not rr:
                continue
            if protocol.endswith('absent'):
                stats = {k: float(np.mean([r[k] for r in rr])) for k in
                         ('false_positive_area', 'any_activation', 'area_activation_1pct')}
                stats['episodes'] = len(rr)
            else:
                stats = present_summary(rr)
            stats['per_class'] = {}
            for c in sorted({r['target_class'] for r in rr}):
                cc = [r for r in rr if r['target_class'] == c]
                stats['per_class'][str(c)] = ({'episodes': len(cc), 'false_positive_area':
                    float(np.mean([r['false_positive_area'] for r in cc]))} if protocol.endswith('absent') else present_summary(cc))
            output[protocol][model] = stats
    return output


def paired_intervals(rows):
    local = {(r['position'], r['model']): r for r in rows if r['protocol'] == 'local'}
    report = {}
    for protocol in ('same_slide_present', 'cross_slide_present'):
        report[protocol] = {}
        for model in ('j5', 'dense'):
            rr = [r for r in rows if r['protocol'] == protocol and r['model'] == model]
            if not rr:
                continue
            groups = sorted({r['query_spatial_group'] for r in rr})
            membership = [np.array([i for i, r in enumerate(rr) if r['query_spatial_group'] == g]) for g in groups]
            rng = np.random.default_rng(20261016)
            draws = rng.integers(0, len(groups), (2000, len(groups)))
            stats = {'paired_episodes': len(rr), 'query_spatial_blocks': len(groups)}
            for metric in ('dice', 'bf1_2', 'unprompted_dice'):
                # Boundary/unclicked scores can be undefined; compare only pairs
                # with an evaluable score on both sides, never impute zero.
                delta = np.array([r[metric] - local[(r['position'], model)][metric]
                                  if r[metric] is not None and local[(r['position'], model)][metric] is not None
                                  else np.nan for r in rr], dtype=float)
                finite = np.isfinite(delta)
                if not finite.any():
                    stats[metric] = {'status': 'no evaluable paired tasks'}; continue
                eligible = [m[finite[m]] for m in membership if finite[m].any()]
                draws = rng.integers(0, len(eligible), (2000, len(eligible)))
                sums = np.array([delta[m].sum() for m in eligible]); n = np.array([len(m) for m in eligible])
                bootstrap = sums[draws].sum(1) / n[draws].sum(1)
                stats[metric] = {'remote_minus_local': float(delta[finite].mean()), 'evaluable_pairs': int(finite.sum()),
                                 'spatial_block_bootstrap_95': np.percentile(bootstrap, [2.5, 97.5]).tolist()}
            report[protocol][model] = stats
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1024)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if ROOT not in args.output.resolve().parents or not 1 <= args.count <= 1024:
        raise ValueError('artifact root/count')
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ('encoded', 'masks'):
        (args.output/folder).mkdir()
    started = time.time(); torch.set_num_threads(4); device = torch.device('cuda:0')
    source = json.loads((DENSE_RUN/'run.json').read_text())
    original = json.loads((Path(source['source_cache'])/'run.json').read_text())
    metadata = json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
    data_args = argparse.Namespace(**original['args'])
    data_args.spatial_data = Path(data_args.spatial_data); data_args.routing = Path(data_args.routing)
    dataset = make_loader(data_args, metadata, 'val').dataset.dataset
    spatial = pd.read_parquet(data_args.spatial_data/'spatial_manifest.parquet').set_index('patch_id')
    patches = pd.read_parquet(data_args.spatial_data/'patches_index.parquet').set_index('patch_id')
    refs = source['refs']['val'][:args.count]
    # Freeze pairing before looking at any model prediction.
    pairs, skipped = build_pairs(dataset.episodes, refs, spatial, patches)
    if args.smoke:
        # Cover every available target class, not only the first background task.
        positions = {}
        for pair in pairs:
            if pair['protocol'] == 'local':
                positions.setdefault(pair['target_class'], pair['position'])
        selected = set(positions.values())
        pairs = [p for p in pairs if p['position'] in selected]
        if not all(any(p['protocol'] == mode for p in pairs) for mode in
                   ('local', 'same_slide_present', 'same_slide_absent', 'cross_slide_present')):
            raise ValueError('smoke must cover all four protocols')
    checkpoint = DENSE_RUN/'dense128_epoch_031.pth'
    run = {'worker_pid': os.getpid(), 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(), 'smoke': args.smoke, 'test_used': False,
           'scope': 'developmental CRC val; historical encoder exposure; GT-simulated centroid prompts',
           'reference_run': str(DENSE_RUN), 'dense_checkpoint': str(checkpoint),
           'dense_sha256': sha256(checkpoint), 'baseline_checkpoint': source['checkpoint'],
           'pairs': pairs, 'skipped': skipped, 'threshold': .5,
           'requested_query_count': args.count, 'train': False,
           'source_sha256': {str(module_source(n)): sha256(module_source(n)) for n in
                             ('support_query_dense', 'evaluate_dense_transfer')},
           'coordinate_policy': 'support-local xy only; remote query heatmaps all zero',
           'score_policy': 'full valid pixels plus paired exclusion of local prompted online regions',
           'expected_shift': 'dense was trained with local prompts; remote inputs are diagnostic OOD'}
    (args.output/'run.json').write_text(json.dumps(run, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'loading', 'worker_pid': os.getpid()}))
    j5 = load_baseline(metadata, Path(source['checkpoint']), device)
    dense = DenseContextModel().to(device).eval().requires_grad_(False)
    dense.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'], strict=True)
    cache = OrderedDict(); feature_files = {}; replay = []; rows = []; excludes = {}
    encoding_seconds = 0.; decoding_seconds = 0.; smoke_checks = {}
    def fetch(index):
        nonlocal encoding_seconds
        index = int(index); pid = str(dataset.episodes.patch_ids[index])
        if index in cache:
            cache.move_to_end(index); return cache[index]
        begin = time.time()
        batch = collate_joint_pixel_episodes([dataset[index]], max_cells=255)
        prompts = {k: batch[k] for k in PROMPT_KEYS}
        if pid in feature_files:
            encoded = torch.load(feature_files[pid], map_location='cpu', weights_only=False)
        else:
            inputs = move_batch({k: batch[k] for k in IMAGE_KEYS}, device)
            encoded = {k: v.cpu() for k, v in encode_image(j5, inputs).items()}
            path = args.output/'encoded'/f'{len(feature_files):05d}.pth'
            torch.save(encoded, path); feature_files[pid] = str(path)
        result = (encoded, prompts, batch)
        cache[index] = result
        if len(cache) > 8:
            cache.popitem(last=False)
        encoding_seconds += time.time()-begin
        return result
    def on_device(values):
        return {k: v.to(device) for k, v in values.items()}
    for number, pair in enumerate(pairs):
        qe, qp, qb = fetch(pair['query_index']); se, sp, sb = fetch(pair['support_index'])
        pair['support_prompts'] = {k: sp[k].tolist() for k in PROMPT_KEYS}
        qe = on_device(qe); se = on_device(se); prompts = on_device(sp)
        if pair['protocol'] != 'same_slide_absent':
            if int(qb['target_class'][0]) != pair['target_class'] or int(sb['target_class'][0]) != pair['target_class']:
                raise ValueError('requested class mismatch in present task')
            if any(int(qp[s+'_mask'].sum()) != int(sp[s+'_mask'].sum()) for s in ('positive', 'negative')):
                raise ValueError('support annotation budget mismatch')
        truth = qb['pixel_gt'].to(device) == pair['target_class']; valid = qb['pixel_gt'].to(device) != 255
        if pair['protocol'].endswith('absent'):
            if (truth & valid).any():
                raise ValueError('requested class occurs in absent query')
        elif not (truth & valid).any():
            raise ValueError('present query has no target')
        begin = time.time()
        with torch.inference_mode():
            task = encode_support(j5, se, prompts)
            prediction, features = predict_query(j5, dense, qe, task, prompts,
                        query_id=pair['query_id'], support_id=pair['support_id'])
            if pair['protocol'] != 'local' and features['prompt_maps'].count_nonzero():
                raise ValueError('source click leaked into remote query heatmap')
            if pair['protocol'] == 'local':
                masks = [remap_prompt_tokens(qe['tokens'], qe['xy'], on_device(qp)[s+'_xy'],
                                             on_device(qp)[s+'_mask'])[2] for s in ('positive', 'negative')]
                low = qe['assignment_low']; assignment = F.interpolate(low, size=truth.shape[-2:], mode='bilinear', align_corners=False)
                exclusion = (masks[0] | masks[1]).gather(1, assignment.argmax(1).flatten(1)).reshape_as(truth)
                excludes[pair['position']] = exclusion.cpu()
                ref = refs[pair['position']]
                old = torch.load(ref['file'], map_location=device, weights_only=False)
                errors = {'position': pair['position'], 'j5_max_error': float((old['base']-prediction['j5']).abs().max()),
                          'dense_max_error': float((dense(old)['pixel_probability']-prediction['dense']).abs().max())}
                if max(errors['j5_max_error'], errors['dense_max_error']) > 2e-6:
                    raise RuntimeError(f'local replay mismatch {errors}')
                replay.append(errors)
                if args.smoke and not smoke_checks:
                    direct = j5(move_batch(qb, device))
                    direct_error = float((direct['pixel_probability']-prediction['j5']).abs().max())
                    if direct_error > 2e-6:
                        raise RuntimeError('GT-free bridge differs from original J5')
                    # Query GT and target identity never enter the bridge functions.
                    altered = dict(qb); altered['pixel_gt'] = torch.full_like(qb['pixel_gt'], 255)
                    altered['target_class'] = torch.zeros_like(qb['target_class'])
                    repeat = encode_image(j5, move_batch({k: altered[k] for k in IMAGE_KEYS}, device))
                    independent_error = max(float((qe[k].float()-repeat[k].float()).abs().max()) for k in qe)
                    if independent_error:
                        raise RuntimeError('image encoding changed under query GT mutation')
                    path = args.output/'smoke_dense_roundtrip.pth'; torch.save({'model': dense.state_dict()}, path)
                    reloaded = DenseContextModel().to(device).eval()
                    reloaded.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'], strict=True)
                    reload_error = float((reloaded(features)['pixel_probability']-prediction['dense']).abs().max())
                    loss = float(dense_loss(dense(features), {'truth': truth, 'valid': valid}))
                    if reload_error or not np.isfinite(loss):
                        raise RuntimeError('reload/loss smoke failed')
                    smoke_checks.update(direct_j5_max_error=direct_error, query_gt_mutation_error=independent_error,
                                        reload_max_error=reload_error, real_dense_loss=loss)
        decoding_seconds += time.time()-begin
        arrays = {'truth': truth.cpu().numpy()[0], 'valid': valid.cpu().numpy()[0]}
        for name, probability in prediction.items():
            base = {k: pair[k] for k in ('position', 'protocol', 'query_id', 'support_id', 'target_class', 'query_spatial_group')}
            base['mask_file'] = f'masks/{number:05d}.npz'
            if pair['protocol'].endswith('absent'):
                predicted = (probability >= .5) & valid
                area = float(predicted.sum()/valid.sum().clamp_min(1))
                row = base | {'model': name, 'false_positive_area': area, 'any_activation': bool(predicted.any()),
                              'area_activation_1pct': area >= .01, 'valid_pixels': int(valid.sum())}
            else:
                score = valid & ~excludes[pair['position']].to(device)
                full = metrics(probability, truth, valid)
                unprompted = metrics(probability, truth, score) if (truth & score).any() else {'dice': None, 'bf1_2': None}
                row = base | {'model': name} | full | {'unprompted_dice': unprompted['dice'],
                         'unprompted_bf1_2': unprompted['bf1_2'], 'unprompted_valid_pixels': int(score.sum())}
                arrays['unprompted_valid'] = score.cpu().numpy()[0]
            if not torch.isfinite(probability).all():
                raise FloatingPointError('prediction')
            arrays[name] = (probability >= .5).cpu().numpy()[0]
            rows.append(row)
            with (args.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
        np.savez_compressed(args.output/'masks'/f'{number:05d}.npz', **arrays)
        verify_saved_masks(args.output/'masks'/f'{number:05d}.npz', rows[-2:])
        with (args.output/'completed.jsonl').open('a') as f:
            f.write(json.dumps({'number': number, **pair})+'\n')
        (args.output/'status.json').write_text(json.dumps({'status': 'evaluating', 'completed': number+1,
                          'pairs': len(pairs), 'worker_pid': os.getpid(), 'elapsed_seconds': time.time()-started}))
        print('PAIR', number+1, '/', len(pairs), pair['protocol'], 'seconds', round(time.time()-started, 1), flush=True)
        del qe, se, prediction, features
    report = {'scope': run['scope'], 'test_used': False, 'smoke': args.smoke,
              'summary': summarize_rows(rows), 'paired_intervals': paired_intervals(rows),
              'local_replay': {'episodes': len(replay), 'max_j5_error': max(r['j5_max_error'] for r in replay),
                               'max_dense_error': max(r['dense_max_error'] for r in replay)},
              'smoke_checks': smoke_checks, 'pair_counts': pd.Series([p['protocol'] for p in pairs]).value_counts().to_dict(),
              'skipped_counts': pd.Series([p['protocol'] for p in skipped], dtype=str).value_counts().to_dict(),
              'encoded_patches': len(feature_files), 'saved_mask_metrics_verified': True, 'encoding_and_io_seconds': encoding_seconds,
              'decoding_and_smoke_seconds': decoding_seconds, 'elapsed_seconds': time.time()-started}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    (args.output/'feature_index.json').write_text(json.dumps(feature_files, indent=2))
    (args.output/'status.json').write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid()}))
    print('COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
