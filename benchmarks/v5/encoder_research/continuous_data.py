"""SHA-bound image/cell inputs, with scoring/training GT access kept separate."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .cell_catalog import load_cell_catalog


TASKS = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400')
REMOTE_TASKS = Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/transfer_1025_131856/tasks_1025_132300')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget_plan(plan, budget):
    positive, negative = map(int, budget.split('+'))
    assert len(plan['positive_xy']) >= positive and len(plan['negative_xy']) >= negative
    return dict(plan, positive_xy=plan['positive_xy'][:positive], negative_xy=plan['negative_xy'][:negative])


class ContinuousData:
    def __init__(self, images, grids):
        self.images, self.grids = Path(images), Path(grids)
        self.catalog, self.rows = load_cell_catalog(TASKS)
        image_proof = json.loads((self.images / 'acceptance.json').read_text())
        grid_proof = json.loads((self.grids / 'independent_acceptance.json').read_text())
        assert image_proof['status'] == 'accepted_all1737_spatial_features' and image_proof['arm'] == 'phikon'
        assert grid_proof['status'] == 'accepted_train_only1737_cell_grids'
        assert image_proof['index_sha256'] == sha(self.images / 'index.jsonl')
        assert grid_proof['index_sha256'] == sha(self.grids / 'index.jsonl')
        self.image_rows = {r['tile_id']: r for r in map(json.loads, (self.images / 'index.jsonl').read_text().splitlines())}
        self.grid_rows = {r['tile_id']: r for r in map(json.loads, (self.grids / 'index.jsonl').read_text().splitlines())}
        assert self.image_rows.keys() == self.grid_rows.keys() == self.rows.keys()
        self.identity = dict(image_index_sha256=image_proof['index_sha256'], cell_index_sha256=grid_proof['index_sha256'],
                             transforms_sha256=grid_proof['transforms_sha256'])

    @lru_cache(maxsize=64)
    def inputs(self, identity):
        image, grid = self.image_rows[identity], self.grid_rows[identity]
        feature = np.load(image['feature'], allow_pickle=False)
        assert hashlib.sha256(feature.tobytes()).hexdigest() == image['array_sha256']
        assert sha(image['image']) == image['image_sha256']
        assert json.loads(self.rows[identity]['variants_json'])['original']['image_sha256'] == image['image_sha256']
        assert sha(grid['file']) == grid['file_sha256']
        with Image.open(image['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        with np.load(grid['file'], allow_pickle=False) as f:
            cells = {arm: torch.from_numpy(f[arm + '_grid'].copy())[None] for arm in ['xcell', 'lemon']}
        return torch.from_numpy(feature.astype(np.float32))[None], torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255, cells

    def model_inputs(self, plan, arm, device):
        qf, qr, qc = self.inputs(plan['query_id'])
        sf, _, sc = self.inputs(plan['support_id'])
        cell_arm = 'xcell' if arm == 'image_only' else arm
        for sign in ['positive_xy', 'negative_xy']:
            points = np.asarray(plan[sign])
            assert len(points) and points.shape[1:] == (2,) and ((points > 0) & (points < 1)).all()
        return dict(query_feature=qf.to(device), query_rgb=qr.to(device), support_feature=sf.to(device),
                    positive_xy=torch.tensor([plan['positive_xy']], dtype=torch.float32, device=device),
                    negative_xy=torch.tensor([plan['negative_xy']], dtype=torch.float32, device=device),
                    query_cells=qc[cell_arm].to(device), support_cells=sc[cell_arm].to(device),
                    local=plan['query_id'] == plan['support_id'])

    @lru_cache(maxsize=64)
    def labels(self, identity):
        row = self.rows[identity]
        with Image.open(row['labels']) as im:
            gt = np.asarray(im).copy()
        assert hashlib.sha256(gt.tobytes()).hexdigest() == row['labels_array_sha256']
        return gt, np.asarray(json.loads(row['input_mpp_xy_json']))

    def supervision(self, plan, device):
        assert self.rows[plan['query_id']]['split'] == self.rows[plan['support_id']]['split'] == 'train'
        assert plan['split'] == 'train'
        gt, _ = self.labels(plan['query_id'])
        return torch.from_numpy(gt == plan['target'])[None].to(device), torch.from_numpy(gt != 255)[None].to(device)

    def validation(self):
        local = [p for p in json.loads((TASKS / 'validation_maximum.json').read_text()) if p['prompt_group'] == 0]
        remote = [p for p in json.loads((REMOTE_TASKS / 'validation_maximum.json').read_text()) if p['prompt_group'] == 0]
        assert (len(local), len(remote)) == (302, 935)
        for scope, plans in [('local', local), ('remote', remote)]:
            for budget in ['3+2', '1+1', '5+1']:
                for original in plans:
                    yield scope, budget, budget_plan(original, budget)
