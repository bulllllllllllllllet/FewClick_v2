"""Wait for audited GPU slots, then gate/train/evaluate predeclared mechanisms."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from benchmarks.v5.fair_comparison.audit import digest,write
from benchmarks.v5.mechanism_comparison.models import VARIANTS


QUEUES={1:['m1_xcell','m3_no_hierarchy'],3:['m3d_morphology'],
        4:['m3_none','m3_morphology'],5:['m3_xcell','m3_shared_encoder']}
DEPENDENCIES={1:['sam_ft'],3:['wsi_sam_ft','closed_reference'],
              4:['fewclick_m3d_image'],5:['fewclick_m3d_xcell']}


def occupied_gpus():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,index','--format=csv,noheader,nounits'],text=True)
    ids={row.split(',')[0].strip():int(row.split(',')[1]) for row in text.splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {ids[row.split(',')[0].strip()] for row in apps.splitlines() if row.split(',')[0].strip() in ids}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--realized',type=Path,required=True);p.add_argument('--cpu-gates',type=Path,required=True)
    p.add_argument('--gate-tasks',type=Path,required=True);p.add_argument('--gate-realized',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    started=time.time();active={};done={};failed=[];slots={};queues={k:list(v) for k,v in QUEUES.items()};stamp='_'.join(a.output.name.rsplit('_',2)[-2:])
    roots=[Path('benchmarks/v5/fair_comparison'),Path('benchmarks/v5/mechanism_comparison')]
    snapshot={str(p):digest(p) for root in roots for p in root.glob('*.py')}|{str(Path(__file__)):digest(Path(__file__))}
    try:
        gates=json.loads(a.cpu_gates.read_text())
        if set(gates)!=set(VARIANTS):raise ValueError('incomplete mechanism CPU gate map')
        coordinate=json.loads((a.realized/'summary.json').read_text())
        if coordinate['status']!='complete' or coordinate['tasks']!=40000:raise ValueError('full shared coordinates missing')
        if digest(a.realized/'training_plans.jsonl')!=coordinate['training_plans_sha256']:raise ValueError('coordinate identity changed')
        for variant,path in gates.items():
            result=json.loads((Path(path)/'summary.json').read_text())
            if result['status']!='passed' or not result['fresh_model_optimizer_resume'] or not result['optimizer_parameters_unique']:
                raise ValueError('CPU gate failed '+variant)
        if len({json.loads((Path(path)/'summary.json').read_text())['consumed_sha256'] for path in gates.values()})!=1:raise ValueError('CPU gates used different clicks')
        write(a.output/'contract.json',dict(status='locked',coordinator_pid=os.getpid(),gpu_queues=QUEUES,dependencies=DEPENDENCIES,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=snapshot,
            tasks=str(a.tasks),protocol_sha256=digest(a.tasks/'contract.json'),realized=str(a.realized),
            training_coordinate_sha256=coordinate['training_plans_sha256'],cpu_gates=gates,
            steps=10000,warmup=1000,accumulate=4,eval_every=1000,
            native_gate='four real GPU updates, fresh model/optimizer/RNG resume, then new formal initialization',
            slot_policy='wait for completed dependency train+evaluation and no active GPU compute process; never use0/2',
            failure_policy='no retry or alternative algorithm; fail the affected queue, preserve independent workers'))
        stages={gpu:None for gpu in queues}
        def launch(gpu,variant,stage):
            for path,expected in snapshot.items():
                if digest(path)!=expected:raise ValueError('locked source changed '+path)
            output=a.output/f'{stage}_{variant}_{stamp}';log=output.with_suffix('.log')
            if stage=='evaluate':
                selected=json.loads((a.output/f'train_{variant}_{stamp}'/'best_checkpoint.json').read_text())
                if digest(selected['path'])!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
                command=[sys.executable,'-u','-m','benchmarks.v5.mechanism_comparison.evaluate','--tasks',str(a.tasks),
                    '--checkpoint',selected['path'],'--method',variant,'--output',str(output)]
            else:
                gate=stage=='gate';command=[sys.executable,'-u','-m','benchmarks.v5.mechanism_comparison.train',
                    '--tasks',str(a.gate_tasks if gate else a.tasks),'--realized',str(a.gate_realized if gate else a.realized),
                    '--method',variant,'--output',str(output)]+(['--gate'] if gate else [])
            handle=log.open('x');env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
            worker=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            active[gpu]=(worker,handle,variant,stage,output,log)
            slots[gpu]=dict(status='running',gpu=gpu,variant=variant,stage=stage,worker_pid=worker.pid,output=str(output),log=str(log),command=command)
            stages[gpu]=(variant,stage);print(json.dumps(slots[gpu]),flush=True)
        while active or any(queues.values()):
            try:
                dependency=json.loads((a.dependency/'status.json').read_text())
            except json.JSONDecodeError:
                # The upstream status writer updates in place. A partial read is
                # an observation race, not a failed experiment or retry trigger.
                print(json.dumps(dict(event='upstream_status_write_in_progress',action='observe same status again; no worker restart')),flush=True)
                time.sleep(1)
                continue
            complete=dependency.get('complete',{})
            occupied=occupied_gpus()
            for gpu in queues:
                if gpu in active:continue
                if not queues[gpu]:continue
                keys=[stage+'|'+method for method in DEPENDENCIES[gpu] for stage in ('train','evaluate')]
                if not all(k in complete for k in keys):
                    failed_dependency=[v for v in dependency.get('failed',[]) if v['method'] in DEPENDENCIES[gpu]]
                    if failed_dependency:
                        failed.append(dict(gpu=gpu,reason='failed upstream dependency',evidence=failed_dependency));queues[gpu]=[]
                        slots[gpu]=dict(status='failed_dependency',gpu=gpu,evidence=failed_dependency)
                    else:slots[gpu]=dict(status='waiting_dependency',gpu=gpu,required=keys)
                    continue
                for key in keys:
                    if json.loads((Path(complete[key])/'summary.json').read_text())['status']!='complete':raise ValueError('dependency summary incomplete')
                if gpu in occupied:
                    slots[gpu]=dict(status='waiting_gpu_idle',gpu=gpu);continue
                variant=queues[gpu].pop(0);launch(gpu,variant,'gate')
            for gpu,item in list(active.items()):
                worker,handle,variant,stage,output,log=item;code=worker.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del active[gpu]
                if code:
                    failed.append(dict(gpu=gpu,variant=variant,stage=stage,exit_code=code,log=str(log)));queues[gpu]=[]
                    slots[gpu]=dict(status='failed',gpu=gpu,variant=variant,stage=stage,log=str(log));continue
                result=json.loads((output/'summary.json').read_text())
                if result['status']!='complete':raise ValueError('worker incomplete despite successful exit')
                if stage=='gate' and not all(result[k] for k in ('fresh_gate_resume','checkpoint_reload_exact','nonzero_joint_backbone_gradient')):raise ValueError('native mechanism gate failed')
                if stage=='train' and result['training_consumed_sha256']!=coordinate['training_plans_sha256']:raise ValueError('formal task parity failed')
                done[stage+'|'+variant]=str(output)
                if stage=='gate':launch(gpu,variant,'train')
                elif stage=='train':launch(gpu,variant,'evaluate')
                else:slots[gpu]=dict(status='complete',gpu=gpu,variant=variant,stage=stage,output=str(output))
            write(a.output/'status.json',dict(status='running',coordinator_pid=os.getpid(),workers=list(slots.values()),complete=done,failed=failed,seconds=time.time()-started))
            time.sleep(5)
        if failed:raise RuntimeError('one or more mechanism queues failed; no automatic retry')
        if len(done)!=3*len(VARIANTS):raise ValueError('mechanism coverage incomplete')
        report=a.output/f'comparison_{stamp}'
        runs=[done['evaluate|'+v] for v in VARIANTS]
        for variant in ('fewclick_m3d_image','fewclick_m3d_xcell'):
            upstream=json.loads((a.dependency/'status.json').read_text())['complete']
            runs.append(upstream['evaluate|'+variant])
        subprocess.run([sys.executable,'-m','benchmarks.v5.fair_comparison.report_comparison','--tasks',str(a.tasks),'--runs',*runs,'--output',str(report)],check=True)
        write(a.output/'summary.json',dict(status='complete',coordinator_pid=os.getpid(),complete=done,comparison=str(report),seconds=time.time()-started))
        write(a.output/'status.json',dict(status='complete',coordinator_pid=os.getpid(),complete=done,comparison=str(report)))
    except Exception:
        write(a.output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),workers=list(slots.values()),complete=done,failed=failed,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
