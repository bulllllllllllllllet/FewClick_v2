"""True nested source fields, explicit context isolation, and cached cell inputs."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch

from benchmarks.v5.multidomain_data.prepare import dump, read_rows, sha
from benchmarks.v5.multidomain_training.episodes import EpisodeCorpus, KINDS, KIND_PROBS
from benchmarks.v5.multidomain_training.catalog import disjoint
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor

CATALOG = Path('/nfs-medical3/zyh/fewclick_v2/multidomain_training/highres_1023_025438/catalog_1023_025438')
SOURCES = Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/prepare_1020_174937/catalog_1020_180034/sources.parquet')


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


@lru_cache(maxsize=32)
def image_header(path):
    return pyvips.Image.new_from_file(path, access='random')


def context_geometry(row, source):
    """Return native source boxes and global physical footprint, without GT."""
    mx, my = source['available_mpp_x'], source['available_mpp_y']
    e = json.loads(source['extra_json'])
    ox, oy = e.get('slide_origin_xy', [0, 0]) if row['dataset'] == 'tiger' else [0, 0]
    cx = (row['x0_um'] + row['x1_um']) / (2 * mx) - ox
    cy = (row['y0_um'] + row['y1_um']) / (2 * my) - oy
    w = round((row['x1_um'] - row['x0_um']) / mx)
    h = round((row['y1_um'] - row['y0_um']) / my)
    if row['dataset'] == 'crc':
        im = image_header(source['image']); width, height = im.width, im.height
    else:
        width, height = source['width'], source['height']
    boxes = []
    for factor in (1, 2, 4):
        bw, bh = w * factor, h * factor
        x, y = round(cx - bw / 2), round(cy - bh / 2)
        if x < 0 or y < 0 or x + bw > width or y + bh > height:
            raise ValueError('incomplete_native_context')
        boxes.append([x, y, bw, bh])
    x, y, bw, bh = boxes[-1]
    return dict(native_boxes=boxes, source_image=source['image'], native_mpp=[mx, my],
                context_um=[(x+ox)*mx, (y+oy)*my, (x+ox+bw)*mx, (y+oy+bh)*my])


def build_catalog(output, train_limit=1200, val_limit=300):
    output.mkdir(parents=True, exist_ok=False)
    sources = {r['source_id']: r for r in read_rows(SOURCES)}
    rows = read_rows(CATALOG / 'tiles.parquet')
    candidates = defaultdict(list); excluded = []; counts = Counter()
    for row in rows:
        if row['dataset'] not in ('crc', 'tiger', 'ignite'):
            continue
        try:
            geom = context_geometry(row, sources[row['source_id']])
        except ValueError as exc:
            excluded.append(dict(tile_id=row['tile_id'], reason=str(exc)))
            continue
        # Primary pool requires a support-eligible target; no model scores read.
        if not json.loads(row['eligible_json']):
            excluded.append(dict(tile_id=row['tile_id'], reason='no_eligible_support_target'))
            continue
        row = dict(row, multiscale_json=json.dumps(geom))
        row['variants_json'] = json.dumps({'original': json.loads(row['variants_json'])['original']})
        candidates[row['dataset'], row['split']].append(row)
        counts[row['dataset']+'|'+row['split']] += 1
    active = sorted(d for d in ('crc', 'tiger', 'ignite')
                    if candidates[d, 'train'] and candidates[d, 'val'])
    if not active:
        raise ValueError('no dataset has both train and validation full-context inputs')
    chosen = []
    for dataset in active:
        for split, limit in [('val', val_limit), ('train', train_limit)]:
            pool = sorted(candidates[dataset, split], key=lambda r: sha(('paper1025|'+r['tile_id']).encode()))
            chosen.extend(pool[:limit])
    val_by_space = defaultdict(list)
    for row in chosen:
        if row['split'] == 'val':
            val_by_space[row['space_key']].append(json.loads(row['multiscale_json'])['context_um'])
    kept = []
    for row in chosen:
        box = json.loads(row['multiscale_json'])['context_um']
        if row['split'] == 'train' and any(intersects(box, b) for b in val_by_space[row['space_key']]):
            excluded.append(dict(tile_id=row['tile_id'], reason='expanded_context_cross_split_overlap'))
            continue
        kept.append(row)
    for dataset in active:
        if not any(r['dataset'] == dataset and r['split'] == 'train' for r in kept):
            raise ValueError('context isolation removed entire training domain '+dataset)
    table = pd.DataFrame(kept); table.to_parquet(output/'tiles.parquet', index=False)
    weights = dict(crc=.375, tiger=.3125, ignite=.3125)
    mass = sum(weights[d] for d in active)
    policy = dict(version='paper_true_context_1025', datasets=active,
                  probabilities=[weights[d]/mass for d in active], qualities=['original'],
                  catalog_sha256=sha((output/'tiles.parquet').read_bytes()),
                  context_scope='1x/2x/4x source fields; train/val largest footprints disjoint')
    dump(output/'training_policy.json', policy)
    dump(output/'excluded.json', excluded)
    dump(output/'summary.json', dict(candidate_counts=dict(counts), datasets=active,
         counts=dict(Counter(r['dataset']+'|'+r['split'] for r in kept)),
         excluded_domains=[d for d in ('crc','tiger','ignite') if d not in active],
         source_catalog_sha256=sha((CATALOG/'tiles.parquet').read_bytes()),
         source_metadata_sha256=sha(SOURCES.read_bytes()), historical_initialization_exposure=True,
         test_used=False, puma_used=False, patient_grouping=False))
    return kept


def native_context(row, scale):
    g = json.loads(row['multiscale_json']); box = g['native_boxes'][scale]
    image = image_header(g['source_image']).crop(*box).extract_band(0, n=3)
    return np.frombuffer(image.write_to_memory(), np.uint8).reshape(image.height, image.width, 3).copy()


class MultiscaleCorpus(EpisodeCorpus):
    """Plan only explicitly available protocols; require real cached contexts."""
    def __init__(self, *args, **kwargs):
        catalog=Path(args[0] if args else kwargs['catalog'])
        self.context_boxes=np.array([json.loads(r['multiscale_json'])['context_um']
                                     for r in read_rows(catalog/'tiles.parquet')])
        super().__init__(*args, **kwargs)
        self.protocol_policy = {}
        for split in ('train', 'val'):
            for dataset in self.training_policy['datasets']:
                available = [k for k in KINDS if self.pools[split][dataset][k]]
                if not available:
                    raise ValueError('no feasible protocols '+split+'/'+dataset)
                weights = np.array([KIND_PROBS[KINDS.index(k)] for k in available])
                self.protocol_policy[split, dataset] = (available, weights/weights.sum())

    def candidates(self,support,indices,relation):
        selected=super().candidates(support,indices,relation)
        if relation=='same' and len(selected):
            selected=selected[disjoint(self.context_boxes[support],self.context_boxes[selected])]
        return selected

    def plan(self, number, **kwargs):
        split = kwargs.get('split', 'train'); dataset = kwargs.get('dataset')
        rng = np.random.default_rng(self.seed+int(number)+700000000)
        if dataset is None:
            dataset = str(rng.choice(self.training_policy['datasets'], p=self.training_policy['probabilities']))
        if kwargs.get('kind') is None:
            kinds, probabilities = self.protocol_policy[split, dataset]
            kwargs['kind'] = str(rng.choice(kinds, p=probabilities))
        kwargs['dataset'] = dataset
        return super().plan(number, **kwargs)

    @lru_cache(maxsize=64)
    def inputs(self, index, quality):
        if quality != 'original':
            raise ValueError('paper comparison uses original image quality only')
        result = dict(super().inputs(index, quality)); row = self.rows[index]
        path = Path(row['multiscale_cache'])
        if sha(path.read_bytes()) != row['multiscale_sha256']:
            raise ValueError('context cache identity changed')
        with np.load(path, allow_pickle=False) as archive:
            for scale in (1, 2):
                cells = archive[f'cells_{scale}'].copy()
                result[f'image_{scale}'] = he_to_tensor(archive[f'image_{scale}'].copy())[None]
                result[f'cells_{scale}'] = torch.from_numpy(cells)[None]
                result[f'cell_valid_{scale}'] = torch.ones((1, len(cells)), dtype=torch.bool)
                result[f'total_{scale}'] = torch.tensor([int(archive[f'total_{scale}'])], dtype=torch.float32)
        return result


def validation_plans(corpus, repetitions=12):
    plans = []; number = 20000000
    for dataset in corpus.training_policy['datasets']:
        for kind in corpus.protocol_policy['val', dataset][0]:
            for _ in range(repetitions):
                p = corpus.plan(number, split='val', dataset=dataset, kind=kind,
                                quality_aug=False, fixed_clicks=True)
                plans.append(p | dict(quality_condition='original', pair_id=number)); number += 1
    return plans


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-limit', type=int, default=1200); p.add_argument('--val-limit', type=int, default=300)
    a = p.parse_args(); build_catalog(a.output, a.train_limit, a.val_limit)
    print((a.output/'summary.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
