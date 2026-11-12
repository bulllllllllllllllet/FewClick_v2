"""Four real train groups, exact cached/full, gradient/state and prompt checks."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from .model import DecoderModel, DecoderData, frozen_backbone_state, frozen_decoder_state
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    a = parser.parse_args()
    c = json.loads((a.root/'contract.json').read_text())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    data = DecoderData(c['cache'], a.arm)
    model = DecoderModel(a.arm).eval()
    plans = json.loads(Path(c['functional_source']).read_text())
    groups = sorted({(t['dataset'], t['kind']) for t in plans})
    assert set(groups) == {('crc', 'local'), ('tiger', 'local'),
                           ('crc', 'same_present'), ('crc', 'same_absent')}
    selected = [next(t for t in plans if (t['dataset'], t['kind']) == group) for group in groups]
    initial = tree_sha(model.state_dict())
    frozen = tree_sha(frozen_backbone_state(model))
    decoder_frozen = tree_sha(frozen_decoder_state(model))
    checks = []
    with torch.no_grad():
        for task in selected:
            assert data.rows[task['support_id']]['split'] == data.rows[task['query_id']]['split'] == 'train'
            x = data.model_inputs(task, 'xcell', 'cuda:0')
            se = model.encode(x['support'])
            qe = se if x['shared'] else model.encode(x['query'])
            cached = model.decode(se, qe, x['prompts'], x['local'])['pixel_probability']
            full = model(**x)
            assert torch.equal(cached, full['query']['pixel_probability'])
            assert torch.isfinite(episode_loss(full, data, task, x['prompts']))
            checks.append(dict(dataset=task['dataset'], kind=task['kind'], cached_full_exact=True))
    # New factory returns the exact previously accepted model, without wrapper state.
    with torch.no_grad():
        references = [model(**data.model_inputs(t, 'xcell', 'cuda:0'))['query']['pixel_probability'].cpu()
                      for t in selected]
        fresh = DecoderModel(a.arm).eval()
        assert tree_sha(fresh.state_dict()) == initial
        for task, probability in zip(selected, references):
            assert torch.equal(fresh(**data.model_inputs(task, 'xcell', 'cuda:0'))['query']['pixel_probability'].cpu(), probability)
    del fresh, references
    torch.cuda.empty_cache()
    model.train()
    task = next(t for t in selected if t['kind'] == 'same_present')
    x = data.model_inputs(task, 'xcell', 'cuda:0')
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4, weight_decay=1e-4)
    timings = []
    gradients = []
    for _ in range(3):
        before = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = episode_loss(model(**x), data, task, x['prompts'])
        loss.backward()
        grads = [v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads)
        magnitude = sum(float(v.norm()) for v in grads)
        assert magnitude > 0
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic()-before)
        gradients.append(magnitude)
    model.eval()
    assert tree_sha(frozen_backbone_state(model)) == frozen
    assert tree_sha(frozen_decoder_state(model)) == decoder_frozen
    with torch.no_grad():
        se, qe = model.encode(x['support']), model.encode(x['query'])
        cached = model.decode(se, qe, x['prompts'], False)['pixel_probability']
        assert torch.equal(cached, model(**x)['query']['pixel_probability'])
        swapped = dict(x['prompts'])
        for key in ['xy', 'mask']:
            swapped['positive_'+key], swapped['negative_'+key] = swapped['negative_'+key], swapped['positive_'+key]
        change = float((model.decode(se, qe, swapped, False)['pixel_probability']-cached).abs().mean())
        assert change > 0
    dump(a.root/a.arm/'engineering.json', dict(status='accepted', pid=os.getpid(), arm=a.arm,
         four_train_groups=checks, initial_fresh_exact=True, cached_full_before_after_exact=True,
         frozen_state_unchanged=True, author_decoder_freeze_applicable=a.arm == 'wsi',
         gradient_norms=gradients, single_task_update_seconds=timings,
         support_sign_swap_mean=change, seconds=time.monotonic()-start,
         trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))


if __name__ == '__main__':
    main()
