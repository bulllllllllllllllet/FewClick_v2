"""Reuse the exact WSI raster/fusion/scoring path with accepted SAM weights."""
import argparse
import json
import os
from pathlib import Path
import torch
from benchmarks.v5.whole_slide_once import worker as raster
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.sam_current_fair.resources import acquire


def load(arm, checkpoint, device):
    if arm == 'med2d':
        from benchmarks.v5.sam_med2d_current.model import DecoderModel
    else:
        from benchmarks.v5.sam_current_fair.model import DecoderModel
    model = DecoderModel(arm, device=device)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device)['model'])
    return model.eval()


def encode(model, value, method):
    return model.encode(value)


def decode(model, support, query, points, method, local=False):
    return model.decode(support, query, points, local)['pixel_probability']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=['sam', 'sam2', 'wsi', 'med2d'], required=True)
    p.add_argument('--case-index', type=int, required=True)
    p.add_argument('--phase', choices=['pilot', 'formal', 'extra'], default='pilot')
    p.add_argument('--gate-only', action='store_true')
    a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text())
    for path, digest in c['sources'].items():
        assert sha(path) == digest, path
    source = Path(c['whole_wsi_root'])
    assert sha(source/'cases.json') == c['cases_sha256']
    armroot = Path(c['adaptations'][a.arm])
    acceptance = json.loads((armroot/'formal/acceptance.json').read_text())
    assert acceptance['status'] == 'complete_accepted_training'
    assert acceptance['updates'] == 2375 and acceptance['consumed'] == 9500
    assert acceptance['training_consumed_sha256'] == c['training_consumed_sha256']
    checkpoint = armroot/'formal/checkpoint.pt'
    assert sha(checkpoint) == acceptance['checkpoint_sha256']
    case = json.loads((source/'cases.json').read_text())[a.case_index]
    assert case['eligible'] and not any(case['membership'].values())
    supports = ([next(s for s in case['supports'] if s['target'] == case['pilot_target'])]
                if a.phase == 'pilot' else case['supports'])
    phase = 'engineering' if a.gate_only else a.phase
    out = a.root/a.arm/phase/f'case_{a.case_index:02}'
    out.mkdir(parents=True, exist_ok=False)
    command = dict(pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   device='cuda:0', arm=a.arm, phase=phase, checkpoint=str(checkpoint),
                   checkpoint_sha256=acceptance['checkpoint_sha256'])
    dump(out/'worker.json', command)
    raster.configure()
    # Only this new process is rebound; frozen original worker files remain intact.
    raster.encode, raster.decode = encode, decode
    lease = acquire(c, int(os.environ['CUDA_VISIBLE_DEVICES']), a.arm+'_whole_WSI')
    try:
        model = load(a.arm, checkpoint, 'cuda:0')
        gate = out/'gate'
        gate.mkdir()
        raster.engineering(model, a.arm, case, supports[0], gate)
        if not a.gate_only:
            report = raster.infer(model, a.arm, case, supports, out/'prediction')
            report.update(encoded_query_reused_across_targets=True,
                          checkpoint_sha256=acceptance['checkpoint_sha256'],
                          adaptation_training_consumed_sha256=c['training_consumed_sha256'],
                          limitations='Fixed matched9500 adaptation and frozen transfer bridge; public initialization/input-view differences retained. Audited train/val-unseen slides with historical AAAI test exposure. Physical WSI BF1 pending.')
            dump(out/'prediction/report.json', report)
            receipt = json.loads((out/'prediction/acceptance.json').read_text())
            receipt['report_sha256'] = sha(out/'prediction/report.json')
            dump(out/'prediction/acceptance.json', receipt)
        dump(out/'acceptance.json', dict(status='accepted_engineering' if a.gate_only else 'complete_accepted_WSI',
             **command, cases_sha256=c['cases_sha256'], targets=[s['target'] for s in supports]))
    except Exception as error:
        dump(out/'failure.json', dict(status='failed_stop_no_retry', error=repr(error)))
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    main()
