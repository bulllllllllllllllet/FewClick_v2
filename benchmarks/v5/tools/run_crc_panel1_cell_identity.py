#!/usr/bin/env python3
"""Panel1 single-cell H&E probe; mIF labels are targets, never input features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import (
    load_ctranspath, crop_region, crop_mask, soft_attention,
)

SOURCE = Path('/nfs-medical6/data/SRRSH/crc/panel_1/registration_outputs/17-00538F_FAP/large_context_sift_registration')
WEIGHTS = Path('/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth')
CLASSES = ['Epithelial', 'Blood_endothelial', 'Lymphatic_endothelial', 'CAF', 'SMA_stromal']
PREPROCESS = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def label(value):
    value = value.removesuffix('_S100')
    if value in ('FAP_CAF', 'FAP_SMA_CAF'):
        value = 'CAF'
    return CLASSES.index(value) if value in CLASSES else -1


def spatial_split(x, y):
    # Entire patch footprint retained inside a block; discarded edges separate splits.
    if not (512 <= x % 4096 <= 3072 and 512 <= y % 4096 <= 3072):
        return None
    group = f'{x // 4096}_{y // 4096}'
    bucket = int(digest('20260923/' + group)[:8], 16) % 10
    return ('train' if bucket < 6 else 'val' if bucket < 8 else 'test'), group


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    quotas = dict(zip(('train', 'val', 'test'), a.patches))
    counts = dict.fromkeys(quotas, 0)
    rows, audit, seen, patch_records = [], [], set(), []
    paths = sorted((SOURCE / 'cell_matching/patch_matches').glob('*.csv.gz'), key=lambda p: digest(p.name))
    for path in paths:
        stem = path.name.removesuffix('.csv.gz')
        x, y = map(int, stem.replace('x', '').replace('y', '').split('_'))
        assignment = spatial_split(x, y)
        if assignment is None:
            continue
        split, group = assignment
        if counts[split] >= quotas[split]:
            continue
        transform = json.loads((SOURCE / 'transforms' / f'{stem}.json').read_text())
        if transform.get('status') != 'ok':
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        required = {'accepted_match', 'base_cell_type', 'he_cell_id', 'he_x', 'he_y',
                    'he_aligned_x', 'he_aligned_y', 'mif_x', 'mif_y', 'x', 'y', 'cell_id'}
        if not required.issubset(frame.columns):
            raise ValueError(f'incomplete source schema: {path}')
        classes = frame.base_cell_type.map(label)
        distance = np.hypot(frame.he_aligned_x-frame.mif_x, frame.he_aligned_y-frame.mif_y)
        accepted = frame.accepted_match.astype(str).str.lower().eq('true')
        keep = accepted & (distance <= 4) & (classes >= 0)
        # Avoid padded image content, retaining identical 128px crop footprints.
        keep &= frame.he_x.between(64, 447) & frame.he_y.between(64, 447)
        selected = frame.loc[keep].copy()
        selected['label'] = classes[keep]
        selected['distance'] = distance[keep]
        audit.append({'patch': stem, 'split': split, 'rows': len(frame), 'accepted': int(accepted.sum()), 'eligible': len(selected)})
        if len(selected) < 5:
            continue
        image_path = SOURCE / 'registered_he_patches' / f'{stem}.png'
        mask_path = SOURCE / 'cell_matching/masks' / f'{stem}.npy'
        if Image.open(image_path).size != (512, 512):
            raise ValueError('unexpected image geometry')
        mask = np.load(mask_path, allow_pickle=False)
        if mask.shape != (512, 512):
            raise ValueError('unexpected H&E mask geometry')
        chosen = selected.sample(frac=1, random_state=20260923).head(a.cells_per_patch)
        added = 0
        for cell in chosen.itertuples():
            key = (round(cell.x, 3), round(cell.y, 3))
            if key in seen:
                continue
            target = mask == int(cell.he_cell_id)
            if not target.any():
                raise ValueError(f'missing H&E nucleus {stem}/{cell.he_cell_id}')
            cy, cx = ndimage.center_of_mass(target)
            if np.hypot(cx-cell.he_x, cy-cell.he_y) > .1:
                raise ValueError(f'H&E centroid mismatch: {stem}/{cell.he_cell_id}')
            seen.add(key)
            rows.append(dict(patch=stem, split=split, group=group, he_id=int(cell.he_cell_id),
                mif_id=int(cell.cell_id), he_x=float(cell.he_x), he_y=float(cell.he_y),
                mif_global_x=float(cell.x), mif_global_y=float(cell.y), label=int(cell.label),
                source_label=cell.base_cell_type, distance=float(cell.distance)))
            added += 1
        if added:
            counts[split] += 1
            patch_records.append({'patch': stem, 'split': split, 'group': group,
                'matches_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'he_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
                'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()})
        if counts == quotas:
            break
    data = pd.DataFrame(rows)
    if not len(data) or set(data.split) != set(quotas):
        raise ValueError(f'insufficient spatial split: {counts}')
    data.to_csv(a.output / 'cells.csv', index=False)
    report = {'status': 'prepared', 'case_id': '2019-00538F', 'classes': CLASSES,
        'source': str(SOURCE), 'patch_counts': counts, 'requested': quotas,
        'support': {s: np.bincount(data.loc[data.split == s, 'label'], minlength=5).tolist() for s in quotas},
        'scope': 'single-case spatial diagnostic, matched known identities only',
        'manifest_sha256': hashlib.sha256((a.output/'cells.csv').read_bytes()).hexdigest(),
        'patches': patch_records, 'filter_audit': audit}
    write_json(a.output/'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('patches', 'filter_audit')}), flush=True)


def score(y, pred):
    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per = []
    for c, name in enumerate(CLASSES):
        tp, support, predicted = int(cm[c, c]), int(cm[c].sum()), int(cm[:, c].sum())
        per.append(dict(name=name, support=support, precision=tp/max(1, predicted),
                        recall=tp/max(1, support), f1=2*tp/max(1, support+predicted)))
    return {'macro_f1': float(np.mean([p['f1'] for p in per])),
            'accuracy': float(np.trace(cm)/max(1, cm.sum())), 'per_class': per, 'confusion': cm.tolist()}


def fit(a):
    torch.set_num_threads(8)
    device = torch.device(a.device)
    data = pd.read_csv(a.output/'cells.csv')
    manifest = json.loads((a.output/'manifest.json').read_text())
    if hashlib.sha256((a.output/'cells.csv').read_bytes()).hexdigest() != manifest['manifest_sha256']:
        raise ValueError('manifest changed')
    if (a.output/'features.npz').exists():
        raise FileExistsError('use a fresh experiment directory')
    model = load_ctranspath(WEIGHTS, device).requires_grad_(False)
    feat = np.zeros((len(data), 768), np.float32)
    for patch, frame in data.groupby('patch', sort=False):
        record = next(r for r in manifest['patches'] if r['patch'] == patch)
        image_path = SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path = SOURCE/'cell_matching/masks'/f'{patch}.npy'
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record['he_sha256'] or hashlib.sha256(mask_path.read_bytes()).hexdigest() != record['mask_sha256']:
            raise ValueError('source image/mask changed')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.load(mask_path, allow_pickle=False)
        tensors = []
        for cell in frame.itertuples():
            cx, cy = round(cell.he_x), round(cell.he_y)
            tensors.append(PREPROCESS(soft_attention(crop_region(image, cy, cx), crop_mask(mask, cy, cx) == cell.he_id)))
        with torch.inference_mode():
            values = torch.cat([model(batch.to(device)).cpu() for batch in torch.stack(tensors).split(32)]).numpy()
        feat[frame.index] = values
        print(json.dumps({'event': 'features', 'patch': patch, 'cells': len(frame)}), flush=True)
    if not np.isfinite(feat).all():
        raise ValueError('nonfinite features')
    np.savez_compressed(a.output/'features.npz', features=feat)
    del model
    torch.cuda.empty_cache()
    x = torch.nn.functional.normalize(torch.from_numpy(feat), dim=1)
    y = torch.tensor(data.label.to_numpy(), dtype=torch.long)
    ids = {s: torch.tensor(np.flatnonzero(data.split == s)) for s in ('train', 'val', 'test')}
    counts = torch.bincount(y[ids['train']], minlength=5)
    if (counts == 0).any():
        raise ValueError(f'train misses classes: {counts.tolist()}')
    prototypes = torch.stack([x[ids['train']][y[ids['train']] == c].mean(0) for c in range(5)])
    prototypes = torch.nn.functional.normalize(prototypes, dim=1)
    weights = counts.float().rsqrt(); weights /= weights.mean()
    reports = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        head = torch.nn.Linear(768, 5)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
        best, history = -1., []
        for epoch in range(a.epochs):
            head.train()
            for batch in ids['train'][torch.randperm(len(ids['train']))].split(256):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(head(x[batch]), y[batch], weight=weights)
                loss.backward(); optimizer.step()
            head.eval()
            with torch.no_grad():
                value = score(y[ids['val']].numpy(), head(x[ids['val']]).argmax(1).numpy())
            history.append({'epoch': epoch, 'val': value})
            if value['macro_f1'] > best:
                best = value['macro_f1']
                torch.save({'head': head.state_dict(), 'classes': CLASSES, 'epoch': epoch,
                    'encoder': str(WEIGHTS), 'manifest_sha256': manifest['manifest_sha256'],
                    'crop_size': 128, 'features': 'L2-normalized CTransPath with target HE nucleus soft attention'}, a.output/f'head_{seed}.pth')
        checkpoint = torch.load(a.output/f'head_{seed}.pth', map_location='cpu', weights_only=False)
        head.load_state_dict(checkpoint['head']); head.eval()
        with torch.no_grad():
            probabilities = head(x[ids['test']]).softmax(1).numpy()
        pred = probabilities.argmax(1)
        result = score(y[ids['test']].numpy(), pred)
        table = data.iloc[ids['test'].numpy()].copy(); table['predicted'] = pred
        for c, name in enumerate(CLASSES): table[f'p_{name}'] = probabilities[:, c]
        table.to_csv(a.output/f'test_predictions_{seed}.csv', index=False)
        reports.append({'seed': seed, 'best_epoch': checkpoint['epoch'], 'test': result, 'history': history})
        print(json.dumps({'seed': seed, 'test': result}), flush=True)
    truth = y[ids['test']].numpy()
    baseline = score(truth, np.full(len(truth), int(counts.argmax())))
    prototype_score = score(truth, (x[ids['test']] @ prototypes.T).argmax(1).numpy())
    write_json(a.output/'metrics.json', {'status': 'complete', 'scope': manifest['scope'],
        'classes': CLASSES, 'support': manifest['support'], 'majority': baseline,
        'nearest_train_prototype': prototype_score, 'runs': reports,
        'mean_test_macro_f1': float(np.mean([r['test']['macro_f1'] for r in reports]))})


def predict(a):
    torch.set_num_threads(8)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    image = np.asarray(Image.open(a.image).convert('RGB'))
    mask = np.asarray(Image.open(a.mask)) > 0
    if image.shape != (128, 128, 3) or mask.shape != (128, 128) or not mask.any():
        raise ValueError('require 128x128 H&E crop and nonempty target nucleus binary mask')
    encoder = load_ctranspath(Path(checkpoint['encoder']), torch.device(a.device))
    head = torch.nn.Linear(768, 5); head.load_state_dict(checkpoint['head']); head.eval()
    with torch.inference_mode():
        features = encoder(PREPROCESS(soft_attention(image, mask))[None].to(a.device)).cpu()
        probs = head(torch.nn.functional.normalize(features, dim=1) * checkpoint.get('feature_scale', 1.)).softmax(1)[0].numpy()
    print(json.dumps({'predicted': CLASSES[int(probs.argmax())], 'probabilities': dict(zip(CLASSES, map(float, probs))),
        'scope': 'closed-set, uncalibrated; unsupported identities may be misclassified'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'fit', 'predict'])
    p.add_argument('--output', type=Path)
    p.add_argument('--patches', nargs=3, type=int, default=[96, 32, 32])
    p.add_argument('--cells-per-patch', type=int, default=128)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--checkpoint', type=Path); p.add_argument('--image', type=Path); p.add_argument('--mask', type=Path)
    a = p.parse_args()
    if a.mode != 'predict' and (a.output is None or not a.output.resolve().is_relative_to('/nfs-medical3/zyh/fewclick_v2')):
        p.error('use timestamped FewClick artifact directory')
    if min(*a.patches, a.cells_per_patch, a.epochs) < 1:
        p.error('positive budgets required')
    {'prepare': prepare, 'fit': fit, 'predict': predict}[a.mode](a)


if __name__ == '__main__':
    main()
