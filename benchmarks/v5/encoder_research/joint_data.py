"""Accepted FP32 prefix inputs for matched Phikon tail adaptation."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .cell_catalog import load_cell_catalog
from .continuous_data import ContinuousData, TASKS, sha


class JointData(ContinuousData):
    def __init__(self, prefixes):
        self.prefixes = Path(prefixes)
        receipt = json.loads((self.prefixes / 'independent_acceptance.json').read_text())
        assert receipt['status'] == 'accepted_all1737_frozen23block_prefixes'
        assert receipt['index_sha256'] == sha(self.prefixes / 'index.jsonl')
        assert receipt['contract_sha256'] == sha(self.prefixes / 'contract.json')
        self.catalog, self.rows = load_cell_catalog(TASKS)
        self.prefix_rows = {r['tile_id']: r for r in map(json.loads, (self.prefixes / 'index.jsonl').read_text().splitlines())}
        assert self.prefix_rows.keys() == self.rows.keys()
        self.identity = dict(prefix_index_sha256=receipt['index_sha256'], prefix_contract_sha256=receipt['contract_sha256'])

    @lru_cache(maxsize=64)
    def inputs(self, identity):
        item = self.prefix_rows[identity]
        assert sha(item['file']) == item['file_sha256']
        prefix = np.load(item['file'], allow_pickle=False)
        assert prefix.shape == (1025, 1024) and prefix.dtype == np.float32 and np.isfinite(prefix).all()
        assert hashlib.sha256(prefix.tobytes()).hexdigest() == item['array_sha256']
        assert sha(item['image']) == item['image_sha256']
        assert json.loads(self.rows[identity]['variants_json'])['original']['image_sha256'] == item['image_sha256']
        with Image.open(item['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        return torch.from_numpy(prefix)[None], torch.from_numpy(rgb).permute(2,0,1)[None].float()/255

    def model_inputs(self, plan, tail, device):
        qp, rgb = self.inputs(plan['query_id'])
        sp, _ = self.inputs(plan['support_id'])
        query = tail(qp.to(device))
        local = plan['query_id'] == plan['support_id']
        support = query if local else tail(sp.to(device))
        points = {}
        for sign in ['positive_xy', 'negative_xy']:
            value = np.asarray(plan[sign])
            assert value.ndim == 2 and value.shape[1] == 2 and len(value) and ((value > 0) & (value < 1)).all()
            points[sign] = torch.tensor([plan[sign]], dtype=torch.float32, device=device)
        return dict(query_feature=query, query_rgb=rgb.to(device), support_feature=support, local=local, **points)
