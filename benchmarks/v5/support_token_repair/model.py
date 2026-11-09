"""Add click-local detail to region-pooled support tokens; keep the query path."""
import torch
from torch import nn
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderModel as Parent, DecoderData as Data
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells

ARMS = ('point_detail', 'local_detail')


class DecoderModel(Parent):
    def __init__(self, arm, seed=20261103, device='cuda:0'):
        assert arm in ARMS
        super().__init__('match', seed, device)
        self.arm = arm
        devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 907)
            self.detail_projection = nn.Linear(256, 256).to(device)
            nn.init.zeros_(self.detail_projection.weight)
            nn.init.zeros_(self.detail_projection.bias)

    def detail(self, encoded, xy):
        features = encoded['pixel_embedding'].float()
        point = sample_assignment_at_cells(features, xy.float()).transpose(1, 2)
        if self.arm == 'point_detail':
            return point
        # Fixed 3x3 fine-feature neighbourhood. No predicted mask or GT is used.
        h, w = features.shape[-2:]
        offsets = xy.new_tensor([(x / w, y / h) for y in [-1, 0, 1] for x in [-1, 0, 1]])
        coords = xy[:, :, None, :] + offsets[None, None, :, :]
        valid = ((coords >= 0) & (coords <= 1)).all(-1)
        neighbours = sample_assignment_at_cells(features, coords.flatten(1, 2)).transpose(1, 2)
        neighbours = neighbours.reshape(*xy.shape[:2], 9, 256)
        similarity = (F.normalize(neighbours, dim=-1) * F.normalize(point, dim=-1)[:, :, None]).sum(-1)
        weights = (similarity / 0.1).masked_fill(~valid, float('-inf')).softmax(-1)
        return (neighbours * weights[..., None]).sum(2)

    def support_task(self, encoded, prompts):
        pool = self.base.core.decoder.prompt_model
        values = dict(prompts)
        raw, masks = [], []
        for sign, identity in [('positive', 0), ('negative', 1)]:
            xy, valid = prompts[sign + '_xy'], prompts[sign + '_mask']
            weights = sample_assignment_at_cells(encoded['assignment_low'].float(), xy.float())
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
            region = torch.einsum('bkp,bkd->bpd', weights, encoded['tokens'][0].float())
            token = region + self.detail_projection(self.detail(encoded, xy))
            values[sign + '_tokens'] = token * valid[:, :, None]
            raw.append(pool.prompt_token_norm(token) + pool.prompt_geometry(xy) + pool.sign_embedding.weight[identity])
            masks.append(valid)
        return pool.encode_prompt_task(values) | dict(raw_prompts=torch.cat(raw, 1), raw_valid=torch.cat(masks, 1))

    def cached_decode(self, se, qe, prompts, local, task=None):
        task = self.support_task(se, prompts) if task is None else task
        output, maps = self.base.core.decode(qe, task, prompts, local)
        if self.pixel is not None and self.pixel_enabled:
            output = self.pixel(se['pixel_embedding'], qe['pixel_embedding'], prompts, output)
        return output, maps

    def forward(self, support, query, prompts, local, shared, with_support=True):
        support = self.base.transform(support)
        query = support if shared else self.base.transform(query)
        se = self.encode(support)
        qe = se if shared else self.encode(query)
        task = self.support_task(se, prompts)
        output, maps = self.cached_decode(se, qe, prompts, local, task)
        source = output if shared else self.cached_decode(se, se, prompts, True, task)[0] if with_support else None
        regularization = se['regularization'] if shared else .5 * (se['regularization'] + qe['regularization'])
        return dict(query=output, support=source, query_prompt_maps=maps,
                    task_token=task['task_token'], assignment_regularization=regularization)


class DecoderData(Data):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(cache, 'match', seed)


def frozen_backbone_state(model):
    return {k: v for k, v in model.state_dict().items() if '.backbone.' in k}


def frozen_decoder_state(model):
    return {}  # No frozen decoder; used only as a nonapplicable sentinel.


def build(arm, checkpoint=None):
    model = DecoderModel(arm)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'])
    return model
