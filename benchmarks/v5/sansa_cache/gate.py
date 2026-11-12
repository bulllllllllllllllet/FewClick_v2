import argparse
import json
import os
import time
from pathlib import Path
import torch
from benchmarks.v5.whole_slide_once.worker import configure
from benchmarks.v5.whole_slide_once.data import inputs, prompts, boxes_at
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.sam_current_fair.resources import acquire
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha
from .model import CachedSANSA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text())
    dump(a.root/'gate_worker.json', dict(pid=os.getpid(), gpu=os.environ['CUDA_VISIBLE_DEVICES'], status='waiting_lease'))
    lease = acquire(c, int(os.environ['CUDA_VISIBLE_DEVICES']), 'sansa_cache_equivalence')
    try:
        configure()
        from benchmarks.v5.same_slide_retrain.model import build
        assert sha(c['checkpoint']) == c['checkpoint_sha256']
        original = build('sansa', c['checkpoint']).eval()
        cached = CachedSANSA(original).eval()
        state = tree_sha(original.state_dict())
        cases = json.loads(Path(c['cases']).read_text())
        results = []
        for i, case in enumerate(cases[:3]):
            selected = [next(s for s in case['supports'] if s['target'] == case['pilot_target']), case['supports'][-1]]
            for si, support in enumerate(selected):
                sx = inputs(case['wsi_path'], support['geometry']['native_boxes'])
                points = prompts(support)
                se = cached.encode(sx)
                bw, bh = support['geometry']['native_boxes'][0][2:]
                for kind, box in [('corner', [0, 0, bw, bh]),
                                  ('far_edge', [case['width']-bw, case['height']-bh, bw, bh])]:
                    qx = inputs(case['wsi_path'], boxes_at(*box))
                    torch.cuda.synchronize(); start = time.monotonic()
                    with torch.no_grad():
                        reference = original(sx, qx, points, False, False, False)['query']['pixel_probability']
                    torch.cuda.synchronize(); native_seconds = time.monotonic()-start
                    start = time.monotonic()
                    qe = cached.encode(qx)
                    actual = cached.decode(se, qe, points)['pixel_probability']
                    torch.cuda.synchronize(); cold_seconds = time.monotonic()-start
                    start = time.monotonic()
                    repeated = cached.decode(se, qe, points)['pixel_probability']
                    torch.cuda.synchronize(); warm_seconds = time.monotonic()-start
                    error = float((reference-actual).abs().max())
                    row = dict(case=i, support=si, query=kind, max_error=error,
                         exact=torch.equal(reference, actual), repeat_exact=torch.equal(actual, repeated),
                         binary_exact=torch.equal(reference >= .5, actual >= .5),
                         native_seconds=native_seconds, cached_cold_seconds=cold_seconds, warm_decode_seconds=warm_seconds)
                    results.append(row)
                    dump(a.root/'progress.json', dict(results=results, status='equivalence_testing'))
                    assert row['exact'] and row['repeat_exact'] and row['binary_exact'], row
                # The native one-frame local path is retained exactly.
                with torch.no_grad():
                    reference = original(sx, sx, points, True, True, False)['query']['pixel_probability']
                assert torch.equal(reference, cached.decode(se, se, points, True)['pixel_probability'])
                swapped = dict(points, positive_xy=points['negative_xy'], positive_mask=points['negative_mask'],
                               negative_xy=points['positive_xy'], negative_mask=points['positive_mask'])
                with torch.no_grad():
                    reference = original(sx, qx, swapped, False, False, False)['query']['pixel_probability']
                assert torch.equal(reference, cached.decode(se, qe, swapped)['pixel_probability'])
            cached.support_memory.clear()
        assert tree_sha(original.state_dict()) == state
        dump(a.root/'acceptance.json', dict(status='accepted_exact_cache_equivalence', results=results,
             remote_cases=12, local_cases=6, swapped_prompt_cases=6, weights_unchanged=True,
             checkpoint_sha256=c['checkpoint_sha256'], no_query_memory_accumulation=True))
    except Exception as error:
        dump(a.root/'failure.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise
    finally:
        lease.close()


if __name__ == '__main__':
    main()
