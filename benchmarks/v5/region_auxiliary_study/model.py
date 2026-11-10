"""Expose the existing pre-matcher region predictions for training supervision."""
import torch
from benchmarks.v5.support_token_repair.model import DecoderModel as Detail, DecoderData as Data
from benchmarks.v5.support_token_repair.model import frozen_backbone_state, frozen_decoder_state

ARMS = ('aux_original', 'aux_detail')


class DecoderModel(Detail):
    def __init__(self, arm, seed=20261103, device='cuda:0'):
        assert arm in ARMS
        super().__init__('point_detail', seed, device)
        self.arm = arm
        if arm == 'aux_original':
            del self.detail_projection

    def detail(self, encoded, xy):
        from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
        return sample_assignment_at_cells(encoded['pixel_embedding'].float(), xy.float()).transpose(1, 2)

    def support_task(self, encoded, prompts):
        if self.arm == 'aux_original':
            return self.base.core.support_task(encoded, prompts)
        return super().support_task(encoded, prompts)

    def forward(self, support, query, prompts, local, shared, with_support=True):
        support = self.base.transform(support)
        query = support if shared else self.base.transform(query)
        se = self.encode(support)
        qe = se if shared else self.encode(query)
        task = self.support_task(se, prompts)
        region, maps = self.base.core.decode(qe, task, prompts, local)
        source_region = region if shared else self.base.core.decode(se, task, prompts, True)[0] if with_support else None
        output = self.pixel(se['pixel_embedding'], qe['pixel_embedding'], prompts, region)
        source = output if shared else self.pixel(se['pixel_embedding'], se['pixel_embedding'], prompts, source_region) if with_support else None
        regularization = se['regularization'] if shared else .5 * (se['regularization'] + qe['regularization'])
        return dict(query=output, support=source, region_query=region, region_support=source_region,
                    query_prompt_maps=maps, task_token=task['task_token'], assignment_regularization=regularization)


class DecoderData(Data):
    def __init__(self, cache, arm, seed=20261103):
        assert arm in ARMS
        super().__init__(cache, 'point_detail', seed)


def build(arm, checkpoint=None):
    model = DecoderModel(arm)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cuda:0')['model'])
    return model
