"""Explicit GPU queues for gated local adaptation and frozen-checkpoint evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from .adapted_models import METHODS
from .audit import digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--realized',type=Path,required=True)
    p.add_argument('--gates',type=Path,required=True,help='JSON mapping method to successful gate directory')
    p.add_argument('--verified',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();processes={};failed=[];complete={};phase={}
    try:
        gates=json.loads(a.gates.read_text());verified=json.loads(a.verified.read_text())
        if verified['status']!='passed' or set(verified['methods'])!=set(METHODS):raise ValueError('independent gate audit incomplete')
        realized=json.loads((a.realized/'summary.json').read_text())
        if realized['status']!='complete' or not realized['complete_protocol'] or realized['tasks']!=40000:raise ValueError('training coordinates incomplete')
        if realized['training_plans_sha256']!=digest(a.realized/'training_plans.jsonl'):raise ValueError('realized training changed')
        for method in METHODS:
            gate=Path(gates[method]);summary=json.loads((gate/'summary.json').read_text());run=json.loads((gate/'run.json').read_text())
            if summary['status']!='complete' or not summary['fresh_gate_resume'] or not summary['checkpoint_reload_exact']:raise ValueError('method gate incomplete')
            for source,expected in run['contract']['source_sha256'].items():
                if digest(source)!=expected:raise ValueError('source changed since real training gate: '+source)
        snapshot={str(path):digest(path) for path in Path(__file__).parent.glob('*.py')}
        write(a.output/'contract.json',dict(status='locked',coordinator_pid=os.getpid(),tasks=str(a.tasks),realized=str(a.realized),
            protocol_sha256=digest(a.tasks/'contract.json'),training_coordinate_sha256=realized['training_plans_sha256'],
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=snapshot,
            gates=gates,independent_audit=str(a.verified),steps=10000,warmup=1000,accumulate=4,eval_every=1000,
            gpu_queues={'1':['sam_ft'],'3':['wsi_sam_ft','closed_reference'],'4':['fewclick_m3d_image'],'5':['fewclick_m3d_xcell']},
            environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version()),
            failure_policy='record failure; no retry or algorithm substitution; independent queues continue'))
        stamp=a.output.name.rsplit('_',2)[-2:];stamp='_'.join(stamp)
        queues={1:[('train','sam_ft'),('evaluate','sam_ft')],
                3:[('train','wsi_sam_ft'),('train','closed_reference'),('evaluate','wsi_sam_ft'),('evaluate','closed_reference')],
                4:[('train','fewclick_m3d_image'),('evaluate','fewclick_m3d_image')],
                5:[('train','fewclick_m3d_xcell'),('evaluate','fewclick_m3d_xcell')]}
        train_runs={m:a.output/f'train_{m}_{stamp}' for m in METHODS}
        evaluation_runs={m:a.output/f'evaluate_{m}_{stamp}' for m in METHODS}
        def launch(gpu):
            if not queues[gpu]:return
            kind,method=queues[gpu].pop(0)
            for source,expected in snapshot.items():
                if digest(source)!=expected:raise ValueError('pipeline source changed before launching '+method)
            output=train_runs[method] if kind=='train' else evaluation_runs[method]
            if kind=='train':
                command=[sys.executable,'-u','-m','benchmarks.v5.fair_comparison.train_adapted','--tasks',str(a.tasks),
                    '--realized',str(a.realized),'--method',method,'--output',str(output)]
            else:
                selected=json.loads((train_runs[method]/'best_checkpoint.json').read_text())
                if digest(selected['path'])!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
                command=[sys.executable,'-u','-m','benchmarks.v5.fair_comparison.evaluate_checkpoint','--tasks',str(a.tasks),
                    '--checkpoint',selected['path'],'--method',method,'--output',str(output)]
            log=output.with_suffix('.log');handle=log.open('x')
            env=os.environ|dict(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')
            process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
            processes[gpu]=(process,handle,kind,method,output,log)
            phase[gpu]=dict(status='running',kind=kind,method=method,worker_pid=process.pid,gpu=gpu,output=str(output),log=str(log),command=command)
            print(json.dumps(phase[gpu]),flush=True)
        for gpu in queues:launch(gpu)
        while processes:
            for gpu,item in list(processes.items()):
                process,handle,kind,method,output,log=item;code=process.poll()
                if code is None:continue
                handle.write(f'\nEXIT_CODE={code}\n');handle.close();del processes[gpu]
                if code!=0:
                    failed.append(dict(kind=kind,method=method,exit_code=code,log=str(log)))
                    phase[gpu]=dict(status='failed',kind=kind,method=method,exit_code=code,log=str(log));queues[gpu]=[]
                    continue
                result=json.loads((output/'summary.json').read_text())
                if result['status']!='complete':raise ValueError('successful process has incomplete summary')
                if kind=='train' and (not result['full_manifest_parity'] or result['training_consumed_sha256']!=realized['training_plans_sha256']):raise ValueError('training coordinate parity failed')
                complete[kind+'|'+method]=str(output);phase[gpu]=dict(status='complete',kind=kind,method=method,output=str(output));launch(gpu)
            write(a.output/'status.json',dict(status='running',coordinator_pid=os.getpid(),workers=list(phase.values()),complete=complete,failed=failed,seconds=time.time()-started))
            time.sleep(3)
        if failed:raise RuntimeError('adaptation queue failed; inspect retained logs')
        if len(complete)!=2*len(METHODS):raise ValueError('incomplete adaptation/evaluation coverage')
        report=a.output/f'comparison_{stamp}'
        command=[sys.executable,'-m','benchmarks.v5.fair_comparison.report_comparison','--tasks',str(a.tasks),'--runs',
            *[str(evaluation_runs[m]) for m in METHODS],'--output',str(report)]
        subprocess.run(command,check=True)
        write(a.output/'summary.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete,comparison=str(report),seconds=time.time()-started))
        write(a.output/'status.json',dict(status='complete',coordinator_pid=os.getpid(),complete=complete,comparison=str(report)))
    except Exception:
        write(a.output/'status.json',dict(status='failed',coordinator_pid=os.getpid(),workers=list(phase.values()),complete=complete,failed=failed,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
