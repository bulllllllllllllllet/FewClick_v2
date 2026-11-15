import argparse
import json
from pathlib import Path
import shutil
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True); a = p.parse_args()
    c = json.loads((a.root/'contract.json').read_text()); formal = a.root/a.arm/'formal'
    acc = json.loads((formal/'acceptance.json').read_text())
    assert acc['status'] == 'complete_accepted_training' and acc['training_consumed_sha256'] == c['training_consumption_sha256']
    out = a.root/a.arm/'evaluation'; out.mkdir()
    for name in ['tasks.json', 'query_rows.json', 'sampling.json']: shutil.copy2(Path(c['sample'])/name, out/name)
    tasks = json.loads((out/'tasks.json').read_text()); kinds = json.loads((out/'sampling.json').read_text())['task_strata']
    gate = []
    for kind in ['small', 'large', 'absent']:
        gate.extend([i for i, t in enumerate(tasks) if kinds[str(t['index'])]['stratum'] == kind][:2])
    assert len(gate) == 6
    dump(out/'contract.json', dict(method=a.arm, tasks_sha256=sha(out/'tasks.json'), query_rows_sha256=sha(out/'query_rows.json'),
        sources=c['sources'], cache=c['cache'], gate_indices=gate, threshold=.5,
        checkpoints={a.arm:dict(path=str(formal/'checkpoint.pt'), sha256=acc['checkpoint_sha256'])},
        comparison_track='Matched9500/2375 frozen1100 stratified development subset; signed support3+2, zero query human clicks'))


if __name__ == '__main__': main()
