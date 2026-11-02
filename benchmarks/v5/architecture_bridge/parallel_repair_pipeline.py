"""Explicitly resharded repair, exact inventory merge and unchanged downstream gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    contract=json.loads((r/'manifest.json').read_text());old=Path(contract['old_root']);py=sys.executable
    def check_sources():
        for path,digest in contract['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    def status(value):
        (r/'stage.json').write_text(json.dumps(value,indent=2))
        with (r/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def command(i,accept=False):
        return [py,'-u','-m','benchmarks.v5.architecture_bridge.sharded_repair','--manifest',str(r/'manifest.json'),'--shard',str(i),'--output',str(r/f'shard{i}'),'--gate-receipt',str(old/'gate/acceptance.json')]+(['--accept'] if accept else [])
    workers=[]
    for i,gpu in enumerate(contract['GPUs']):
        check_sources();log=(r/f'shard{i}.log').open('w');cmd=command(i)
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        workers.append((proc,log));status(dict(status='running',stage='extraction',shard=i,GPU=gpu,worker_pid=proc.pid,command=cmd,time=time.time()))
    failed=False
    for i,(proc,log) in enumerate(workers):
        rc=proc.wait();log.write('\nEXIT_CODE='+str(rc)+'\n');log.close();failed=failed or rc!=0
    if failed:status(dict(status='failed',stage='extraction',retry=False));return
    for i in range(len(workers)):
        check_sources()
        with (r/f'shard{i}_accept.log').open('w') as log:
            rc=subprocess.call(command(i,True),stdout=log,stderr=subprocess.STDOUT)
        if rc:status(dict(status='failed',stage='shard_accept',shard=i));return
    from .repair_cell_cache import verify_geometry
    from benchmarks.v5.encoder_research.continuous_data import sha
    from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
    from benchmarks.v5.encoder_research.continuous_data import TASKS
    from benchmarks.v4.phase_3_cell_region.src.cells import encode_xcell_features
    import numpy as np
    _,inventory=load_cell_catalog(TASKS);proof=json.loads((old/'gate/acceptance.json').read_text())
    frames=contract['adopted'][:]
    for i in range(len(workers)):
        report=json.loads((r/f'shard{i}/report.json').read_text());assert report['identity']==proof['identity'];frames+=report['frames']
    assert len(frames)==len(inventory)==1737 and len({x['tile_id'] for x in frames})==1737
    frames.sort(key=lambda x:x['tile_id']);assert {x['tile_id'] for x in frames}==inventory.keys()
    status(dict(status='running',stage='merged_geometry_accept',worker_pid=os.getpid()))
    for row in frames:
        assert sha(row['file'])==row['sha256']
        assert row['split']==inventory[row['tile_id']]['split'] and row['dataset']==inventory[row['tile_id']]['dataset']
        with np.load(row['file']) as f:
            for s in range(3):
                verify_geometry(f[f'instances_{s}'],f[f'meta_{s}'],f[f'ids_{s}'],row['tile_id']+'|'+str(s),f[f'mpp_{s}'])
                assert np.array_equal(f[f'cells_{s}'],encode_xcell_features(f[f'meta_{s}'],f[f'reg_{s}'])) and np.isfinite(f[f'cells_{s}']).all()
    merged=r/'merged';merged.mkdir(exist_ok=False)
    (merged/'report.json').write_text(json.dumps(dict(identity=proof['identity'],frames=frames,gate=False,explicit_reshard_manifest_sha256=sha(r/'manifest.json')),indent=2))
    (merged/'acceptance.json').write_text(json.dumps(dict(status='accepted_repaired_raw_cells',report_sha256=sha(merged/'report.json'),identity=proof['identity'],frames=1737,normalization_accepted=False,training_allowed=False)))
    for name,cmd in [('normalize',[py,'-u','-m','benchmarks.v5.architecture_bridge.normalize_repaired_cells','--raw',str(merged),'--output',str(r/'normalized')]),('functional',[py,'-u','-m','benchmarks.v5.architecture_bridge.functional_repaired','--cache',str(r/'normalized'),'--output',str(r/'functional')])]:
        check_sources()
        with (r/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ|dict(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            status(dict(status='running',stage=name,worker_pid=proc.pid,command=cmd));rc=proc.wait();log.write('\nEXIT_CODE='+str(rc)+'\n')
        if rc:status(dict(status='failed',stage=name,retry=False));return
    status(dict(status='complete',formal_P3_training_started=False))


if __name__=='__main__':main()
