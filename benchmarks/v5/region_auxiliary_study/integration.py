"""Real six-group initialization, cached inference and learnable-injection checks."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from .model import DecoderModel, DecoderData, ARMS
from benchmarks.v5.support_match_study.model import DecoderModel as Baseline
from .loss import episode_loss
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss as original_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    a = p.parse_args()
    c = json.load(open(a.root / 'contract.json'))
    for path, digest in c['sources'].items():
        assert sha(path) == digest, path
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plans = list(map(json.loads, Path(c['training_source']).read_text().splitlines()))
    selected = [next(t for t in plans if t['dataset'] == d and t['kind'] == k)
                for d in ['crc', 'tiger'] for k in ['local', 'cross_present', 'cross_absent']]
    data = DecoderData(Path(c['cache']), a.arm)
    assert all(data.rows[t['support_id']]['split'] == data.rows[t['query_id']]['split'] == 'train' for t in selected)
    from benchmarks.v5.support_token_repair.model import DecoderModel as DetailBaseline
    baseline = (Baseline('match') if a.arm=='aux_original' else DetailBaseline('point_detail')).eval()
    baseline_sha = tree_sha(baseline.state_dict())
    with torch.no_grad():
        expected = [baseline(**data.model_inputs(t, 'xcell', 'cuda:0'))['query']['pixel_probability'].cpu() for t in selected]
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    model = DecoderModel(a.arm).eval()
    assert tree_sha(model.state_dict()) == baseline_sha
    with torch.no_grad():
        for task, reference in zip(selected, expected):
            inputs = data.model_inputs(task, 'xcell', 'cuda:0')
            prediction = model(**inputs)
            assert torch.isfinite(episode_loss(prediction, data, task, inputs['prompts']))
            actual = prediction['query']['pixel_probability']
            assert torch.equal(actual.cpu(), reference)
            se = model.encode(model.base.transform(inputs['support']))
            qe = se if inputs['shared'] else model.encode(model.base.transform(inputs['query']))
            cached = model.cached_decode(se, qe, inputs['prompts'], inputs['local'])[0]['pixel_probability']
            assert torch.equal(actual, cached)
    model.train()
    inputs = data.model_inputs(selected[0], 'xcell', 'cuda:0')
    # Isolate the auxiliary gradient: it must train region logits, not matcher parameters.
    prediction = model(**inputs)
    original = original_loss(prediction, data, selected[0], inputs['prompts'])
    total = episode_loss(prediction, data, selected[0], inputs['prompts'])
    # Inspect the auxiliary graph directly; subtracting two full loss graphs
    # introduces floating cancellation residuals at the shared matcher output.
    region = dict(prediction, query=prediction['region_query'], support=prediction['region_support'])
    auxiliary = original_loss(region, data, selected[0], inputs['prompts']) - prediction['assignment_regularization']
    assert torch.equal(total, original + .5 * auxiliary)
    assert torch.isfinite(total) and auxiliary > 0
    cancelled_gradients = torch.autograd.grad((total-original)/.5, list(model.pixel.parameters()), retain_graph=True, allow_unused=True)
    cancellation_residual = max(float(g.abs().max()) for g in cancelled_gradients if g is not None)
    auxiliary.backward()
    grad = model.base.core.decoder.delta_head[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(v.grad is None or not v.grad.any() for v in model.pixel.parameters())
    auxiliary_gradient_norm = float(grad.norm())
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4, weight_decay=1e-4)
    gradient_norms = []
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = episode_loss(model(**inputs), data, selected[0], inputs['prompts'])
        assert torch.isfinite(loss)
        loss.backward()
        grad = model.base.core.decoder.delta_head[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        gradient_norms.append(float(grad.norm()))
        opt.step()
    model.eval()
    with torch.no_grad():
        se = model.encode(model.base.transform(inputs['support']))
        cached = model.cached_decode(se, se, inputs['prompts'], True)[0]['pixel_probability']
        actual = model(**inputs)['query']['pixel_probability']
        assert inputs['shared'] and torch.equal(cached, actual)
    dump(a.root / a.arm / 'integration.json', dict(status='complete_accepted_engineering', worker_pid=os.getpid(),
         common_initial_state_sha256=baseline_sha, six_groups_initial_output_exact=True,
         cached_full_exact_before_and_after_update=True, region_head_gradient_norms=gradient_norms, auxiliary_region_gradient_norm=auxiliary_gradient_norm, auxiliary_matcher_gradient_zero=True, subtracted_graph_cancellation_residual_max=cancellation_residual,
         additional_parameters_vs_corresponding_control=0, auxiliary_weight=.5,
         tasks=selected, train_only=True, formal_initialization='fresh constructor; engineering weights discarded'))


if __name__ == '__main__':
    main()
