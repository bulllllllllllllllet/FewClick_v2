"""Frozen common state and parent-mask references for the one cell-only revision."""
import json
from pathlib import Path

import numpy as np
import torch

from .continuous_data import sha
from .continuous_prompt import tree_sha


def freeze_shared(model):
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith('cell_project.'))
    assert sum(v.numel() for v in model.parameters() if v.requires_grad) == 76800


def assert_shared_frozen(model, optimizer, parent):
    current = model.state_dict()
    for name, value in parent['model'].items():
        assert torch.equal(current[name].detach().cpu(), value), name
    for name, parameter in model.named_parameters():
        if not name.startswith('cell_project.'):
            assert not parameter.requires_grad and parameter.grad is None, name
    state = optimizer.state_dict()
    group = state['param_groups'][0]
    shared = dict(state={p: state['state'][p] for p in group['params']}, param_groups=[group])
    assert tree_sha(shared) == tree_sha(parent['optimizer'])


def reference_key(row):
    return tuple(row[key] for key in ['scope', 'budget', 'support_id', 'query_id', 'target'])


class ParentMaskReference:
    def __init__(self, parent_checkpoint, remote_directory):
        local = Path(parent_checkpoint).parent
        remote = Path(remote_directory)
        local_receipt = json.loads((local / 'independent_acceptance.json').read_text())
        remote_receipt = json.loads((remote / 'independent_acceptance.json').read_text())
        assert local_receipt['status'] == 'accepted_fixed_image_pilot'
        assert remote_receipt['status'] == 'accepted_full935_transfer'
        checkpoint_sha = sha(parent_checkpoint)
        assert remote_receipt['checkpoint_sha256'] == checkpoint_sha
        assert json.loads((local / 'report.json').read_text())['checkpoint_sha256'] == checkpoint_sha
        self.masks = {}
        self.identity = {'checkpoint_sha256': checkpoint_sha}
        for scope, directory in [('local', local), ('remote', remote)]:
            rows = json.loads((directory / 'rows.json').read_text())
            with np.load(directory / 'packed_masks.npz', allow_pickle=False) as file:
                masks = file['masks'].copy()
            for row in rows:
                if scope == 'local' and row['scope'] != 'local':
                    continue
                entry = dict(row, scope=scope)
                key = reference_key(entry)
                assert key not in self.masks
                self.masks[key] = masks[row['mask_index']]
            for name in ['independent_acceptance.json', 'rows.json', 'packed_masks.npz']:
                self.identity[scope + '|' + name] = sha(directory / name)
        assert len(self.masks) == 3711

    def assert_equal(self, row, packed):
        assert np.array_equal(self.masks[reference_key(row)], packed), 'frozen image-only prediction differs from parent'
