"""GT-free, independently encoded support/query bridge for frozen J5 + dense.

Support xy always belongs to its source image. Remote query heatmaps are empty.
The bridge preserves the original J5 centroid-prompt semantics, including padding.
"""
import torch
import torch.nn.functional as F
from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import assignment_geometry, remap_prompt_tokens
from benchmarks.v4.phase_6_mask_decoder.src.model import project_region_probabilities
from .dense_prompt_decoder import prompt_maps


@torch.no_grad()
def encode_image(model, inputs):
    if model.cross_scale_variant != 'fine_only' or model.parent_context is not None:
        raise ValueError('bridge requires the selected fine_only J5 checkpoint')
    shallow = {}
    hook = model.phase2.backbone.layer1.register_forward_hook(lambda m, i, o: shallow.update(value=o))
    try:
        features = model.phase2.extract_backbone_features(inputs['image'])
    finally:
        hook.remove()
    p2 = model.phase2.forward_from_features(features, inputs['image'].shape[-2:],
                                            return_full_assignment=False, return_tokens=True)
    mass = sample_assignment_at_cells(p2['assignment_low'], inputs['cells'])
    tokens = model.cell(p2['region_tokens'], mass, inputs['cells'], inputs['cell_valid'],
                        inputs['total_cell_count'].float())['fused_tokens']
    xy, area = assignment_geometry(p2['assignment_low'])
    value = shallow['value']
    value = value.reshape(value.shape[0], 32, 8, *value.shape[-2:]).mean(2)
    return {'assignment_low': p2['assignment_low'], 'tokens': tokens, 'xy': xy, 'area': area,
            'active': inputs.get('fine_active', area > 1e-6),
            'deep': p2['embedding'].half(), 'shallow': value.half(),
            'semantic': F.interpolate(p2['semantic_logits'], size=(64, 64), mode='bilinear',
                                      align_corners=False).half()}


@torch.no_grad()
def encode_support(model, encoded, prompts):
    values = {'prompt_size_id': prompts['prompt_size_id']}
    for sign in ('positive', 'negative'):
        values[sign+'_tokens'] = remap_prompt_tokens(encoded['tokens'], encoded['xy'],
                            prompts[sign+'_xy'], prompts[sign+'_mask'])[0]
        for suffix in ('xy', 'mask'):
            values[sign+'_'+suffix] = prompts[sign+'_'+suffix]
    return model.decoder.prompt_model.encode_prompt_task(values)


def query_features(model, encoded, task, prompts, *, query_id, support_id, size=(512, 512)):
    batch = {'fine_tokens': encoded['tokens'], 'region_xy': encoded['xy'],
             'region_area': encoded['area'], 'fine_active': encoded['active']}
    matched = model.decoder.prompt_model.match_regions(batch, task)
    decoded = model.decoder.refine(batch, matched)
    assignment = F.interpolate(encoded['assignment_low'], size=size, mode='bilinear', align_corners=False)
    assignment = assignment / assignment.sum(1, keepdim=True).clamp_min(1e-6)
    base = project_region_probabilities(assignment, decoded['logits'])['pixel_probability']
    maps = prompt_maps(prompts) if query_id == support_id else base.new_zeros(base.shape[0], 2, 128, 128)
    return {name: encoded[name] for name in ('deep', 'shallow', 'semantic')} | {
        'base': base, 'task': task['task_token'], 'prompt_maps': maps.half()}


@torch.no_grad()
def predict_query(model, dense, encoded, task, prompts, *, query_id, support_id):
    features = query_features(model, encoded, task, prompts, query_id=query_id, support_id=support_id)
    return {'j5': features['base'], 'dense': dense(features)['pixel_probability']}, features
