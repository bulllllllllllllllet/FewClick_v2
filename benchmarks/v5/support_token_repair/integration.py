"""Real six-group initialization, cached inference and learnable-injection checks."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from .model import DecoderModel, DecoderData, ARMS
from benchmarks.v5.support_match_study.model import DecoderModel as Baseline
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
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
    baseline = Baseline('match').eval()
    baseline_sha = tree_sha(baseline.state_dict())
    with torch.no_grad():
        expected = [baseline(**data.model_inputs(t, 'xcell', 'cuda:0'))['query']['pixel_probability'].cpu() for t in selected]
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    model = DecoderModel(a.arm).eval()
    assert tree_sha({k: v for k, v in model.state_dict().items() if not k.startswith('detail_projection.')}) == baseline_sha
    assert not torch.count_nonzero(model.detail_projection.weight).item()
    with torch.no_grad():
        for task, reference in zip(selected, expected):
            inputs = data.model_inputs(task, 'xcell', 'cuda:0')
            actual = model(**inputs)['query']['pixel_probability']
            assert torch.equal(actual.cpu(), reference)
            se = model.encode(model.base.transform(inputs['support']))
            qe = se if inputs['shared'] else model.encode(model.base.transform(inputs['query']))
            cached = model.cached_decode(se, qe, inputs['prompts'], inputs['local'])[0]['pixel_probability']
            assert torch.equal(actual, cached)
            for sign in ['positive', 'negative']:
                detail = model.detail(se, inputs['prompts'][sign + '_xy'])
                assert detail.shape == (*inputs['prompts'][sign + '_xy'].shape[:2], 256)
                assert torch.isfinite(detail).all()
    model.train()
    inputs = data.model_inputs(selected[0], 'xcell', 'cuda:0')
    # The original graph output head is zero-initialized. Its upstream prompt
    # gradients are legitimately zero on update1; require flow after it opens.
    assert not torch.count_nonzero(model.base.core.decoder.delta_head[-1].weight).item()
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4, weight_decay=1e-4)
    gradient_norms = []
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        loss = episode_loss(model(**inputs), data, selected[0], inputs['prompts'])
        assert torch.isfinite(loss)
        loss.backward()
        grad = model.detail_projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        gradient_norms.append(float(grad.norm()))
        opt.step()
    assert gradient_norms[0] == 0 and gradient_norms[-1] > 0
    model.eval()
    with torch.no_grad():
        se = model.encode(model.base.transform(inputs['support']))
        injected = model.support_task(se, inputs['prompts'])['raw_prompts']
        original = model.base.core.support_task(se, inputs['prompts'])['raw_prompts']
        assert not torch.equal(injected, original), 'injection never reaches prompt decoder'
        cached = model.cached_decode(se, se, inputs['prompts'], True)[0]['pixel_probability']
        actual = model(**inputs)['query']['pixel_probability']
        assert inputs['shared'] and torch.equal(cached, actual)
    dump(a.root / a.arm / 'integration.json', dict(status='complete_accepted_engineering', worker_pid=os.getpid(),
         common_initial_state_sha256=baseline_sha, six_groups_initial_output_exact=True,
         cached_full_exact_before_and_after_update=True, new_projection_gradient_norms=gradient_norms,
         projection_changes_prompt_tokens=True, added_trainable_parameters=sum(v.numel() for v in model.detail_projection.parameters()),
         tasks=selected, train_only=True, formal_initialization='fresh constructor; engineering weights discarded'))


if __name__ == '__main__':
    main()
