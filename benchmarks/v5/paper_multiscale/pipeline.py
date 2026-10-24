"""Persisted preparation/training coordinator; failures stop without retries."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pyvips
import pandas as pd

from .data import MultiscaleCorpus,validation_plans
from .train import ARMS
from benchmarks.v5.multidomain_data.prepare import dump,read_rows,sha


def launch(command,output,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',VIPS_CONCURRENCY='4')
    handle=output.with_suffix('.log').open('w')
    worker=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
    return dict(worker=worker,log=handle,command=command,gpu=gpu,output=str(output))


def wait_jobs(jobs,root,stage):
    dump(root/f'{stage}_workers.json',[dict(worker_pid=j['worker'].pid,command=j['command'],gpu=j['gpu'],output=j['output']) for j in jobs])
    live=list(jobs);failed=[]
    while live:
        for job in list(live):
            code=job['worker'].poll()
            if code is None:continue
            job['log'].write(f'\nEXIT_CODE={code}\n');job['log'].close();live.remove(job)
            if code:failed.append(dict(pid=job['worker'].pid,code=code,output=job['output']))
        dump(root/'status.json',dict(status=stage,coordinator_pid=os.getpid(),
             active_workers=[j['worker'].pid for j in live],failed=failed))
        if live:time.sleep(3)
    if failed:raise RuntimeError(f'{stage} failed: {failed}')


def assemble(original,parts,output,allow_subset=False):
    originals={r['tile_id']:r for r in read_rows(original/'tiles.parquet')}
    rows=[]
    for part in parts:
        for row in read_rows(part/'tiles.parquet'):
            old=originals[row['tile_id']]
            for key in ('multiscale_json','variants_json','split','labels_array_sha256'):
                if row[key]!=old[key]:raise ValueError('prepared input identity mismatch '+key)
            if sha(Path(row['multiscale_cache']).read_bytes())!=row['multiscale_sha256']:
                raise ValueError('prepared context hash mismatch')
            rows.append(row)
    ids=[r['tile_id'] for r in rows]
    if len(set(ids))!=len(ids) or (not allow_subset and set(ids)!=set(originals)):
        raise ValueError('prepared shards missing or duplicate rows')
    output.mkdir(parents=True,exist_ok=False)
    # Preserve source ordering because EpisodeCorpus indexes by row position.
    order={key:i for i,key in enumerate(originals)};rows.sort(key=lambda r:order[r['tile_id']])
    pd.DataFrame(rows).to_parquet(output/'tiles.parquet',index=False)
    policy=json.loads((original/'training_policy.json').read_text())
    policy['catalog_sha256']=sha((output/'tiles.parquet').read_bytes());dump(output/'training_policy.json',policy)
    dump(output/'summary.json',dict(counts=dict(Counter(r['dataset']+'|'+r['split'] for r in rows)),
         cache_hashes_verified=True,source_catalog=str(original),parts=[str(p) for p in parts],subset=allow_subset))
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['gate','formal'],required=True)
    p.add_argument('--gpus',default='1,3,4,5');p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--stamp',required=True)
    a=p.parse_args();root=a.root;gpus=list(map(int,a.gpus.split(',')));stamp=a.stamp
    source=root/'catalog_1025_033600';prepared=root/'context_gate_1025_035900'
    started=time.time();command=[sys.executable,'-u','-m','benchmarks.v5.paper_multiscale.train']
    source_identity={str(f):sha(f.read_bytes()) for f in sorted(Path(__file__).parent.glob('*.py'))}
    try:
        if a.stage=='gate':
            if json.loads((prepared/'status.json').read_text())['status']!='complete':raise ValueError('representative context preparation has not passed')
            catalog=assemble(source,[prepared],root/f'gate_catalog_{stamp}',True)
            corpus=MultiscaleCorpus(catalog);plans=validation_plans(corpus,2)
            path=root/f'gate_plans_{stamp}.json';dump(path,plans)
            for arm in ARMS:
                out=root/f'gate_{arm}_{stamp}'
                common=['--catalog',str(catalog),'--arm',arm,'--steps','6','--warmup','2',
                        '--accumulate','1','--eval-every','3','--validation-plans',str(path),'--gate']
                first=launch(command+common+['--output',str(out),'--stop-after','3'],out,gpus[0])
                wait_jobs([first],root,'gate_train_'+arm)
                checkpoint=out/'checkpoint_step_00003.pth'
                resumed=root/f'gate_resume_{arm}_{stamp}'
                second=launch(command+common+['--output',str(resumed),'--resume',str(checkpoint)],resumed,gpus[0])
                wait_jobs([second],root,'gate_resume_'+arm)
                result=json.loads((resumed/'summary.json').read_text())
                if result['status']!='complete' or not result['checkpoint_reload_exact']:raise ValueError('gate failed '+arm)
            dump(root/'gate_acceptance.json',dict(passed=True,stamp=stamp,arms=list(ARMS),
                 real_train_validation_checkpoint_resume=True,seconds=time.time()-started))
        else:
            if not json.loads((root/'gate_acceptance.json').read_text())['passed']:raise ValueError('training gates have not passed')
            if not json.loads((root/'forward_audit_1025_040500.json').read_text())['passed']:raise ValueError('real forward gates have not passed')
            dump(root/f'source_snapshot_{stamp}.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=source_identity))
            jobs=[];parts=[prepared]
            for i,gpu in enumerate(gpus):
                out=root/f'contexts_shard{i:02d}_{stamp}';parts.append(out)
                args=[sys.executable,'-u','-m','benchmarks.v5.paper_multiscale.prepare','--catalog',str(source),
                      '--output',str(out),'--shard',str(i),'--shards',str(len(gpus)),
                      '--exclude-prepared',str(prepared)]
                jobs.append(launch(args,out,gpu))
            wait_jobs(jobs,root,'prepare_full_contexts')
            if any(sha(Path(f).read_bytes())!=value for f,value in source_identity.items()):
                raise ValueError('source changed during context preparation; refusing mixed-version training')
            catalog=assemble(source,parts,root/f'training_catalog_{stamp}')
            corpus=MultiscaleCorpus(catalog);plans=validation_plans(corpus,12)
            plan_path=root/f'validation_plans_{stamp}.json';dump(plan_path,plans)
            dump(root/f'formal_contract_{stamp}.json',dict(catalog=str(catalog),validation_plans=str(plan_path),
                 steps=a.steps,warmup=a.steps//10,global_batch_per_arm=4,arms=list(ARMS),
                 contexts=list(corpus.training_policy['datasets']),all_contexts_disjoint_from_other_split=True))
            if len(gpus)<len(ARMS):raise ValueError('one independent GPU required per matched arm')
            jobs=[]
            for gpu,arm in zip(gpus,ARMS):
                out=root/f'{arm}_{stamp}'
                args=command+['--catalog',str(catalog),'--output',str(out),'--arm',arm,
                     '--steps',str(a.steps),'--warmup',str(a.steps//10),'--accumulate','4',
                     '--eval-every',str(max(1,a.steps//10)),'--validation-plans',str(plan_path)]
                jobs.append(launch(args,out,gpu))
            wait_jobs(jobs,root,'training_matched_arms')
            summaries={arm:json.loads((root/f'{arm}_{stamp}'/'summary.json').read_text()) for arm in ARMS}
            dump(root/f'comparison_{stamp}.json',summaries)
            jobs=[]
            for gpu,arm in zip(gpus,ARMS):
                out=root/f'prompt_budgets_{arm}_{stamp}'
                args=[sys.executable,'-u','-m','benchmarks.v5.paper_multiscale.evaluate_budgets',
                      '--catalog',str(catalog),'--run',str(root/f'{arm}_{stamp}'),'--arm',arm,'--output',str(out)]
                jobs.append(launch(args,out,gpu))
            wait_jobs(jobs,root,'evaluate_nested_prompt_budgets')
        dump(root/'status.json',dict(status=a.stage+'_complete',coordinator_pid=os.getpid(),seconds=time.time()-started))
    except Exception as exc:
        dump(root/'status.json',dict(status='failed',stage=a.stage,coordinator_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
