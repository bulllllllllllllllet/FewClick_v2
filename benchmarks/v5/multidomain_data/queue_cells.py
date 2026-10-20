"""Wait for a completed render, then prepare disjoint InstanSeg GPU shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from .prepare import dump,read_rows,sha
from .prepare_cells import balanced_selection,shard_selection
from .fast_nuclei import DEFAULT_INSTANSEG_CHECKPOINT


def code_identity():
    paths=[Path(__file__),Path(__file__).with_name('prepare_cells.py'),Path(__file__).with_name('fast_nuclei.py'),
        Path('benchmarks/v5/phase_6_mask_decoder/diagnostics/external_cell_inputs.py')]
    return {str(p):sha(p.read_bytes()) for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render-root',type=Path,required=True);p.add_argument('--render-log',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(6)));p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit',type=int,default=0,help='0 covers the complete rendered manifest')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_INSTANSEG_CHECKPOINT)
    a=p.parse_args()
    if len(set(a.gpus))!=len(a.gpus) or min(a.gpus)<0 or a.limit<0:raise ValueError('invalid GPU/limit selection')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();jobs=[];logs=[];identity=code_identity()
    def status(state,**extra):
        dump(a.output/'status.json',dict(status=state,worker_pid=os.getpid(),seconds=time.time()-started,**extra))
    try:
        gate=json.loads(a.gate.read_text())
        if gate.get('status')!='passed':raise ValueError('real model smoke gate has not passed')
        dump(a.output/'run.json',dict(render_root=str(a.render_root),render_log=str(a.render_log),gate=str(a.gate),
            gate_sha256=sha(a.gate.read_bytes()),checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint.read_bytes()),
            gpus=a.gpus,threads_per_worker=a.threads,limit=a.limit,source_sha256=identity))
        while True:
            # Render writes EXIT_CODE only after the final manifest/status. Never
            # read a partial manifest or race its non-atomic status-file writes.
            with a.render_log.open('rb') as log:
                log.seek(max(0,a.render_log.stat().st_size-4096));tail=log.read().decode(errors='replace')
            endings=[line for line in tail.splitlines() if line.startswith('EXIT_CODE=')]
            if endings:
                if endings[-1]!='EXIT_CODE=0':raise RuntimeError('upstream render failed: '+endings[-1])
                break
            if time.time()-started>86400:raise TimeoutError('render did not finish within 24 hours')
            status('waiting_for_render',render_root=str(a.render_root));time.sleep(30)
        if code_identity()!=identity:raise ValueError('source changed while queued; refusing mixed code versions')
        render=json.loads((a.render_root/'summary.json').read_text())
        if render.get('status')!='complete' or render.get('test_pixels_read') is not False:raise ValueError('render gate failed')
        inputs=read_rows(a.render_root/'inputs.parquet')
        if len(inputs)!=render['input_variants']:raise ValueError('render count mismatch')
        if any(r['split']=='test' or (r['dataset']=='puma' and r['training_allowed']) for r in inputs):raise ValueError('heldout/split contract changed')
        selected=balanced_selection(inputs,a.limit) if a.limit else inputs
        shard_selection(selected,0,len(a.gpus))  # Validate global identity uniqueness.
        manifest=a.output/'selected_inputs.parquet';pd.DataFrame(selected).to_parquet(manifest,index=False)
        dump(a.output/'selection.json',dict(inputs=len(selected),source_manifest=str(a.render_root/'inputs.parquet'),
            source_manifest_sha256=sha((a.render_root/'inputs.parquet').read_bytes()),selected_sha256=sha(manifest.read_bytes())))
        for index,gpu in enumerate(a.gpus):
            out=a.output/f'shard_{index:02d}';logpath=a.output/f'shard_{index:02d}.log'
            args=[sys.executable,'-u','-m','benchmarks.v5.multidomain_data.prepare_cells','--inputs',str(manifest),
                '--output',str(out),'--all','--detector','instanseg','--instanseg-checkpoint',str(a.checkpoint),
                '--shard-index',str(index),'--num-shards',str(len(a.gpus)),'--threads',str(a.threads)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads))
            handle=logpath.open('w');logs.append(handle);worker=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,env=env)
            jobs.append(dict(worker=worker,output=out,gpu=gpu,log=logpath,command=args))
        dump(a.output/'workers.json',[dict(pid=j['worker'].pid,gpu=j['gpu'],output=str(j['output']),log=str(j['log']),command=j['command']) for j in jobs])
        finished=set()
        while len(finished)<len(jobs):
            for index,j in enumerate(jobs):
                result=j['worker'].poll()
                if result is None or index in finished:continue
                finished.add(index)
                logs[index].write(f'\nEXIT_CODE={result}\n');logs[index].flush()
                if result:raise RuntimeError(f'GPU {j["gpu"]} worker {j["worker"].pid} failed with {result}')
            status('running',inputs=len(selected),completed_shards=len(finished),workers=[j['worker'].pid for j in jobs])
            if len(finished)<len(jobs):time.sleep(5)
        merged=[];recipes=set()
        expected={(r['tile_id'],r['quality']):r for r in selected};seen=set()
        for j in jobs:
            report=json.loads((j['output']/'summary.json').read_text())
            if report.get('status')!='complete' or report.get('detector')!='instanseg':raise ValueError('shard did not complete with InstanSeg')
            recipes.add(report['cell_recipe_id'])
            for row in read_rows(j['output']/'inputs_with_cells.parquet'):
                key=(row['tile_id'],row['quality'])
                if key not in expected or key in seen:raise ValueError('unexpected or duplicate shard identity')
                if row['image_sha256']!=expected[key]['image_sha256'] or row['split']!=expected[key]['split']:raise ValueError('shard source identity changed')
                if row['cell_recipe_id']!=report['cell_recipe_id'] or not Path(row['cells']).is_file():raise ValueError('invalid cell cache identity')
                seen.add(key);merged.append(row)
        if len(recipes)!=1 or seen!=set(expected):raise ValueError('incomplete or mixed-recipe cache')
        pd.DataFrame(merged).to_parquet(a.output/'inputs_with_cells.parquet',index=False)
        dump(a.output/'summary.json',dict(status='complete',inputs=len(merged),cell_recipe_id=next(iter(recipes)),detector='instanseg',
            seconds=time.time()-started,shards=len(jobs),gpus=a.gpus,split_identity_verified=True,all_shards_roundtrip_verified=True,
            tissue_GT_used=False,mean_worker_seconds_per_variant=sum(r['cell_preparation_seconds'] for r in merged)/len(merged)))
        status('complete',inputs=len(merged));print('ALL_CELL_SHARDS_COMPLETE',len(merged),flush=True)
    except Exception as exc:
        for j in jobs:
            if j['worker'].poll() is None:j['worker'].terminate()
        for j in jobs:
            try:j['worker'].wait(timeout=10)
            except subprocess.TimeoutExpired:j['worker'].kill();j['worker'].wait()
        status('failed',error=repr(exc));raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()
