"""Freeze separate WSI/CPU extension while training source contracts stay immutable."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    root = a.root; assert not (root/'wsi_contract.json').exists()
    c = json.loads((root/'contract.json').read_text())
    for f, h in c['sources'].items(): assert sha(f) == h, f
    cache = Path(c['original_wsi_root'])/'cache_extension.json'; original = json.loads(cache.read_text())
    gate = Path(original['gate_acceptance']); assert sha(gate) == original['gate_acceptance_sha256']
    assert json.loads(gate.read_text())['status'] == 'accepted_exact_cache_equivalence'
    for f, h in original['sources'].items(): assert sha(f) == h, f
    files = set(c['sources']) | set(original['sources']) | {str(p) for p in Path('benchmarks/v5/reviewer_sample').glob('*.py')}
    dump(root/'wsi_contract.json', dict(status='authorized_frozen_extension', methods=['HG_mix', 'sansa', 'M', 'no_cross_scale', 'proto_ms'],
         slides=3, sources={f:sha(f) for f in sorted(files)}, sansa_cache_gate=str(gate), sansa_cache_gate_sha256=sha(gate),
         maximum_workers=4, gpus=[0,1,2,4], reference_reproduction_before_fanout=True,
         storage='Lossless window-projected binary masks and unique-pixel fused/owner masks; float64 original probability accumulation',
         training_contract_sha256=sha(root/'contract.json')))
    import shutil
    for f in files-set(c['sources']):
        target = root/'source_snapshot'/f; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, target)
    environment = dict(TMPDIR=str(root/'tmp'), PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:'+str(Path.cwd()),
         OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VIPS_CONCURRENCY='2', VIPS_DISC_THRESHOLD='100m', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name, module in [('wsi', 'wsi_queue'), ('references', 'references')]:
        command = [sys.executable, '-u', '-m', 'benchmarks.v5.reviewer_sample.'+module, '--root', str(root)]
        log = root/(name+'_controller.log'); session = root.name+'_'+name
        shell = shlex.join(['env', *[k+'='+v for k,v in environment.items()], *command])+' > '+shlex.quote(str(log))+' 2>&1; rc=$?; echo EXIT_CODE=$rc >> '+shlex.quote(str(log))+'; exit $rc'
        subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(Path.cwd()),shell], check=True)
        print(json.dumps(dict(session=session, command=command, log=str(log))), flush=True)


if __name__ == '__main__': main()
