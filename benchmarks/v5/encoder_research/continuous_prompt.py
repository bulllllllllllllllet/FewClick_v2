"""Matched continuous-cell residual on the frozen-feature shared prompt decoder."""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from .shared_prompt import SharedPromptDecoder


ARMS = ('image_only', 'xcell', 'lemon')


def tree_sha(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((str(array.dtype), array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for element in item:
                visit(element)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def mask_loss(logits, truth, valid):
    target, weight = truth.float(), valid.float()
    probability = logits.sigmoid()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weight).sum() / weight.sum()
    dice = 1 - (2 * (probability * target * weight).sum() + 1) / ((probability * weight).sum() + (target * weight).sum() + 1)
    return bce + dice


class ContinuousPromptDecoder(SharedPromptDecoder):
    def __init__(self, arm):
        assert arm in ARMS
        super().__init__(1024)
        self.arm = arm
        self.cell_project = nn.Conv2d(75, 1024, 1, bias=False)
        nn.init.zeros_(self.cell_project.weight)

    def forward(self, query_feature, query_rgb, positive_xy, negative_xy,
                support_feature, query_cells, support_cells, local):
        if self.arm == 'image_only':
            query_cells = torch.zeros_like(query_cells)
            support_cells = torch.zeros_like(support_cells)
        query = query_feature + self.cell_project(query_cells)
        support = support_feature + self.cell_project(support_cells)
        return super().forward(query, query_rgb, positive_xy, negative_xy, support_feature=support, local=local)


def base_parameters(model):
    return [value for name, value in model.named_parameters() if not name.startswith('cell_project.')]


def fresh_optimizer(model):
    optimizer = torch.optim.AdamW(base_parameters(model), lr=.001, weight_decay=.0001)
    optimizer.add_param_group({'params': model.cell_project.parameters()})
    return optimizer


def initialize(parent, arm, seed, device):
    torch.manual_seed(seed)
    state = torch.load(parent, map_location='cpu', weights_only=True)
    assert state['updates'] == 2000 and state.get('seed', 20261031) == seed
    model = ContinuousPromptDecoder(arm).to(device)
    missing = model.load_state_dict(state['model'], strict=False)
    assert missing.missing_keys == ['cell_project.weight'] and not missing.unexpected_keys
    original = {k: v for k, v in model.state_dict().items() if not k.startswith('cell_project.')}
    assert tree_sha(original) == tree_sha(state['model'])
    optimizer = torch.optim.AdamW(base_parameters(model), lr=.001, weight_decay=.0001)
    optimizer.load_state_dict(state['optimizer'])
    assert tree_sha(optimizer.state_dict()) == tree_sha(state['optimizer'])
    assert all(int(v['step']) == 2000 for v in optimizer.state.values())
    optimizer.add_param_group({'params': model.cell_project.parameters()})
    assert model.cell_project.weight not in optimizer.state
    identity = dict(start_model_sha256=tree_sha(model.state_dict()), parent_model_sha256=tree_sha(state['model']),
                    parent_optimizer_sha256=tree_sha(state['optimizer']), starting_updates=2000,
                    head_parameters=sum(v.numel() for v in model.parameters()), cell_parameters=76800,
                    qualification='same registered capacity; image-only cell weights have zero gradients and remain zero')
    return model, optimizer, identity
