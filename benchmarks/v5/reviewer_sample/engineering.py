"""Real-input attribution and cached/full equivalence before learning gates."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from .model import DecoderModel, DecoderData, Full, frozen_backbone_state
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.whole_slide_once.worker import configure
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True)
    a = p.parse_args(); c = json.loads((a.root/'contract.json').read_text()); configure(); start = time.monotonic()
    m = DecoderModel(a.arm); ref = Full('HG_mix'); data = DecoderData(Path(c['cache']), a.arm)
    plans = json.loads(Path(c['functional_source']).read_text()); frozen = tree_sha(frozen_backbone_state(m)); checks = []
    if a.arm == 'fixed_grid':
        assert m.state_dict().keys() == ref.state_dict().keys()
        for k, v in m.state_dict().items(): assert torch.equal(v, ref.state_dict()[k]), k
    for domain, kind in sorted({(t['dataset'], t['kind']) for t in plans}):
        t = next(t for t in plans if (t['dataset'], t['kind']) == (domain, kind))
        x = data.model_inputs(t, 'xcell', 'cuda:0')
        with torch.no_grad():
            calls = []
            backbone = m.backbone if a.arm == 'proto_ms' else m.base.core.encoders[1].backbone
            hook = backbone.encoder.register_forward_hook(lambda *args: calls.append(1))
            try: se = m.encode(m.base.transform(x['support']))
            finally: hook.remove()
            assert len(calls) == 7
            if a.arm == 'proto_ms':
                raw = m.raw_features(x['support'])
                for s in range(3):
                    key = 'image'+('' if s == 0 else '_'+str(s))
                    expected = ref.base.core.encoders[s].backbone(x['support'][key])['out']
                    assert torch.equal(raw[s], expected), s
            else:
                sr = ref.encode(ref.base.transform(x['support']))
                assert torch.equal(se['assignment_low'], sr['assignment_low'])
            qe = se if x['shared'] else m.encode(m.base.transform(x['query']))
            cached = m.cached_decode(se, qe, x['prompts'], x['local'])[0]
            full = m(**x)
            assert torch.equal(cached['pixel_probability'], full['query']['pixel_probability'])
            assert torch.isfinite(episode_loss(full, data, t, x['prompts']))
            if a.arm == 'fixed_grid':
                assert torch.equal(full['query']['pixel_probability'], ref(**x)['query']['pixel_probability'])
        checks.append(dict(dataset=domain, kind=kind, encoder_calls=7, raw_backbone_matched=True, cached_full_exact=True))
        print('accepted', domain, kind, flush=True)
    del ref; torch.cuda.empty_cache()
    t = next(t for t in plans if t['kind'] == 'same_present'); x = data.model_inputs(t, 'xcell', 'cuda:0')
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    for _ in range(3):
        opt.zero_grad(set_to_none=True); loss = episode_loss(m(**x), data, t, x['prompts']); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None); opt.step()
    assert tree_sha(frozen_backbone_state(m)) == frozen
    if a.arm == 'fixed_grid':
        for e in m.base.core.encoders:
            assert e.assignment.weight.grad is None and torch.count_nonzero(e.assignment.weight) == 0
            # Fixed soft assignment is content-invariant, including after optimization.
            with torch.no_grad():
                from benchmarks.v4.phase_2_region_encoder.src.model import spatial_region_assignment
                z = torch.zeros(1, 256, 64, 64, device='cuda:0')
                assert torch.equal(spatial_region_assignment(z, e.assignment, e.spatial_anchors, e.spatial_temperature),
                                   spatial_region_assignment(z+1, e.assignment, e.spatial_anchors, e.spatial_temperature))
    else:
        assert all(p.grad is not None and float(p.grad.norm()) > 0 for p in m.projections.parameters())
        assert float(m.head[-1].weight.grad.norm()) > 0
    with torch.no_grad():
        full = m(**x)['query']['pixel_probability']; prompts = dict(x['prompts'])
        for k in ['xy', 'mask']: prompts['positive_'+k], prompts['negative_'+k] = prompts['negative_'+k], prompts['positive_'+k]
        difference = float((full-m(**dict(x, prompts=prompts))['query']['pixel_probability']).abs().mean())
        assert difference > 0
        se = m.encode(m.base.transform(x['support'])); qe = m.encode(m.base.transform(x['query']))
        assert torch.equal(full, m.cached_decode(se, qe, x['prompts'], False, False)[0]['pixel_probability'])
    dump(a.root/a.arm/'engineering.json', dict(status='accepted', pid=os.getpid(), arm=a.arm, checks=checks,
         prompt_swap_mean=difference, frozen_state_unchanged=True, trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
         seconds=time.monotonic()-start))


if __name__ == '__main__': main()
