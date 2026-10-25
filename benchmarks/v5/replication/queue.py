"""Run extra matched-order seeds only after existing per-GPU experiment queues."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.experiment_orchestration.mechanism_queue import occupied_gpus

METHODS={1:'sam_ft',3:'wsi_sam_ft',4:'fewclick_m3d_image',5:'fewclick_m3d_xcell'}
MECHANISMS={1:['m1_xcell','m3_no_hierarchy'],3:['m3d_morphology'],
            4:['m3_none','m3_morphology'],5:['m3_xcell','m3_shared_encoder']}


def observe(path):
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return None  # An in-place status write is not a failed job.


def run(protocol,mechanisms,external,cpu_gate,output):
    output.mkdir(parents=True,exist_ok=False);started=time.time();active={};complete={};failed=[];slots={}
    manifest=json.loads((protocol/'manifest.json').read_text());seeds=sorted(map(int,manifest))
    queues={gpu:list(seeds) for gpu in METHODS}
    files=[Path(__file__),Path(__file__).with_name('prepare.py'),Path(__file__).with_name('gate_cpu.py')]
    files += list(Path('benchmarks/v5/fair_comparison').glob('*.py'))
    files += [Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/multidomain_training/losses.py')]
    snapshot={str(p):digest(p) for p in files};stamp='_'.join(output.name.rsplit('_',2)[-2:])
    try:
        preparation=json.loads((protocol/'summary.json').read_text());gate=json.loads((cpu_gate/'summary.json').read_text())
        if preparation['status']!='complete_preparation' or not preparation['exact_task_and_click_multiset'] or seeds!=[20261026,20261027]:
            raise ValueError('versioned seed orders incomplete')
        if gate['status']!='passed' or gate['protocol_manifest_sha256']!=digest(protocol/'manifest.json'):
            raise ValueError('real CPU reordered model/loss/update gate missing')
        if gate['source_sha256']!=digest(Path(__file__).with_name('gate_cpu.py')) or preparation['source_sha256']!=digest(Path(__file__).with_name('prepare.py')):
            raise ValueError('CPU-gated preparation/model source changed')
        for value in manifest.values():
            if digest(Path(value['tasks'])/'contract.json')!=value['protocol_sha256'] or digest(Path(value['realized'])/'training_plans.jsonl')!=value['coordinate_sha256']:
                raise ValueError('seed protocol/actual coordinates changed')
            if digest(Path(value['gate_realized'])/'training_plans.jsonl')!=value['gate_coordinate_sha256']:
                raise ValueError('seed native gate coordinates changed')
        write(output/'contract.json',dict(coordinator_pid=os.getpid(),methods=METHODS,seeds=seeds,protocol=str(protocol),
            manifest_sha256=digest(protocol/'manifest.json'),cpu_gate=str(cpu_gate),cpu_gate_sha256=digest(cpu_gate/'summary.json'),
            source_sha256=snapshot,mechanism_dependency=str(mechanisms),GPU4_extra_dependency=str(external),
            budgets=dict(steps=10000,accumulate=4,warmup=1000,eval_every=1000),
            stage_order='native four-update gate, NEW initialization full train, selected-checkpoint6040 scores',
            statistics='include seed20261025 plus both extra seeds; report all seeds and mean/std, never best-seed selection',
            RNG_scope='order/new heads/dropout vary; pretrained weights shared; Dense internal initialization fixed by implementation',
            failure_policy='affected GPU queue stops without retry; independent queues continue'))
        def launch(gpu,seed,stage):
            for path,expected in snapshot.items():
                if digest(path)!=expected:raise ValueError('locked repeated-training source changed '+path)
            value=manifest[str(seed)];method=METHODS[gpu]
            if digest(Path(value['tasks'])/'contract.json')!=value['protocol_sha256']:raise ValueError('seed task contract changed')
            destination=output/f'{stage}_{method}_seed{seed}_{stamp}'
            if stage=='evaluate':
                trained=output/f'train_{method}_seed{seed}_{stamp}'
                selected=json.loads((trained/'best_checkpoint.json').read_text())
                if digest(selected['path'])!=selected['checkpoint_sha256']:raise ValueError('seed selected checkpoint changed')
                command=[sys.executable,'-u','-m','benchmarks.v5.fair_comparison.evaluate_checkpoint',
                    '--tasks',value['tasks'],'--method',method,'--checkpoint',selected['path'],'--output',str(destination)]
            else:
                command=[sys.executable,'-u','-m','benchmarks.v5.fair_comparison.train_adapted',
                    '--tasks',value['tasks'],'--realized',value['gate_realized'] if stage=='gate' else value['realized'],
                    '--method',method,'--seed',str(seed),'--output',str(destination)]
                if stage=='gate':command+=['--gate']
            log=destination.with_suffix('.log');handle=log.open('x')
            env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            worker=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(worker,handle,seed,stage,destination,log)
            slots[gpu]=dict(status='running',gpu=gpu,method=method,seed=seed,stage=stage,worker_pid=worker.pid,
                output=str(destination),log=str(log),command=command)
            print(json.dumps(slots[gpu]),flush=True)
        while active or any(queues.values()):
            upstream=observe(mechanisms/'status.json');outside=observe(external/'status.json')
            if upstream is None or outside is None:time.sleep(1);continue
            occupied=occupied_gpus();done=upstream.get('complete',{})
            for gpu in METHODS:
                if gpu in active or not queues[gpu]:continue
                keys=['evaluate|'+variant for variant in MECHANISMS[gpu]]
                dependency_ready=all(key in done for key in keys)
                own_failure=any(item.get('gpu')==gpu for item in upstream.get('failed',[]))
                terminal_missing=upstream['status']=='failed' and not dependency_ready
                if own_failure or terminal_missing or (gpu==4 and outside['status']=='failed'):
                    failed.append(dict(gpu=gpu,stage='dependency',reason='required queue failed'));queues[gpu]=[]
                    slots[gpu]=dict(status='failed_dependency',gpu=gpu);continue
                if not dependency_ready or (gpu==4 and outside['status']!='complete'):
                    slots[gpu]=dict(status='waiting_dependencies',gpu=gpu,methods=METHODS[gpu]);continue
                for key in keys:
                    if json.loads((Path(done[key])/'summary.json').read_text())['status']!='complete':raise ValueError('mechanism completion lacks summary')
                if gpu in occupied:slots[gpu]=dict(status='waiting_gpu_idle',gpu=gpu);continue
                seed=queues[gpu].pop(0);launch(gpu,seed,'gate')
            for gpu,item in list(active.items()):
                worker,handle,seed,stage,destination,log=item;code=worker.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del active[gpu]
                if code:
                    failed.append(dict(gpu=gpu,method=METHODS[gpu],seed=seed,stage=stage,exit_code=code,log=str(log)))
                    queues[gpu]=[];slots[gpu]=dict(status='failed',gpu=gpu,seed=seed,stage=stage);continue
                result=json.loads((destination/'summary.json').read_text());value=manifest[str(seed)]
                if result['status']!='complete':raise ValueError('seed child incomplete despite zero exit')
                if stage=='gate':
                    if not all(result[k] for k in ('fresh_gate_resume','checkpoint_reload_exact','nonzero_joint_backbone_gradient')):
                        raise ValueError('new seed native gate failed')
                    if result['training_consumed_sha256']!=value['gate_coordinate_sha256']:raise ValueError('new seed gate point order mismatch')
                if stage=='train':
                    if result['step']!=10000 or not result['full_manifest_parity'] or result['training_consumed_sha256']!=value['coordinate_sha256']:
                        raise ValueError('new seed full coordinate parity failed')
                    run_record=json.loads((destination/'run.json').read_text())
                    if run_record['contract']['seed']!=seed:raise ValueError('wrong training seed')
                if stage=='evaluate':
                    rows=sum(1 for _ in (destination/'rows.jsonl').open())
                    if rows!=6040:raise ValueError('seed score coverage incomplete')
                complete[f'{stage}|{METHODS[gpu]}|{seed}']=str(destination)
                if stage=='gate':launch(gpu,seed,'train')
                elif stage=='train':launch(gpu,seed,'evaluate')
                else:slots[gpu]=dict(status='complete',gpu=gpu,method=METHODS[gpu],seed=seed,stage=stage)
            write(output/'status.json',dict(status='running',coordinator_pid=os.getpid(),workers=list(slots.values()),
                complete=complete,failed=failed,seconds=time.time()-started))
            time.sleep(5)
        if failed:raise RuntimeError('one or more repeated-training GPU queues failed; outputs retained')
        if len(complete)!=24:raise ValueError('two extra seeds/four methods/gate+train+score coverage incomplete')
        write(output/'summary.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete,
            seconds=time.time()-started,pending='independent full row/seed parity and mean/std report with original seed'))
        write(output/'status.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete))
    except Exception:
        write(output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),workers=list(slots.values()),complete=complete,failed=failed,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--mechanisms',type=Path,required=True)
    p.add_argument('--external',type=Path,required=True);p.add_argument('--cpu-gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.protocol,a.mechanisms,a.external,a.cpu_gate,a.output)
