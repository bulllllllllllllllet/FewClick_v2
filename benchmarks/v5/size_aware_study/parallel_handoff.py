"""Opportunistic independent-arm parallelism, preserving the in-flight worker."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha


def read(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=a.root;out=a.output;s=read(root/'settings.json');contract=read(out/'resource_contract.json')
    assert sha(__file__)==contract['coordinator_sha256']
    def status(**value):
        value.update(observed_at=time.time(),controller_pid=os.getpid())
        tmp=out/'stage.tmp';tmp.write_text(json.dumps(value));tmp.replace(out/'stage.json')
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        return value
    def canonical(value):
        temp=root/'stage.parallel.tmp';temp.write_text(json.dumps(value));temp.replace(root/'stage.json')
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def verify():
        for path,h in s['sources'].items():assert sha(path)==h,path
        assert sha(root/'contract.json')==contract['execution_contract_sha256']
    def args(arm):
        return ['--arm',arm,'--cache',s['cache'],'--protocol',s['protocol'],'--learning',str(root/'learning'),
                '--inventory',s['inventory'],'--output',str(root/arm),'--execution-contract',str(root/'contract.json'),
                '--benchmark-receipt',str(root/'engineering'/(arm+'_benchmark.json')),'--monitor-receipt',s['monitor']]
    def call(label,module,arguments):
        verify();cmd=[sys.executable,'-u','-m',module,*arguments]
        with (out/(label+'.log')).open('x') as log:
            worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            status(status='accepting',stage=label,worker_pid=worker.pid,command=cmd)
            rc=worker.wait();log.write('\nEXIT_CODE='+str(rc)+'\n');assert rc==0,(label,rc)
    def process_state(pid):
        p=Path(f'/proc/{pid}/stat')
        if not p.exists():return None,None
        fields=p.read_text().rsplit(')',1)[1].split()
        return fields[0],int(fields[-1])
    old_controller=None;adopted=False
    try:
        assert not (out/'history.jsonl').exists(),'No automatic restart'
        while True:
            verify();stage=read(root/'stage.json')
            if stage.get('stage')!='precision_formal':
                if stage.get('status') in ['failed','complete_accepted'] or stage.get('stage','').startswith('detail_'):
                    status(status='not_activated',reason='original controller advanced before another allowed GPU became available',original_stage=stage);return
                status(status='waiting_parallel_opportunity',original_stage=stage);time.sleep(10);continue
            free={gpu:int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) for gpu in [1,4,5]}
            available=[g for g in free if free[g]>=12288]
            if not available:status(status='waiting_GPU_memory',free_MiB=free,minimum_free_MiB=12288,existing_worker_pid=stage['worker_pid']);time.sleep(10);continue
            gpu=max(available,key=lambda g:free[g]);pid=stage['controller_pid'];worker_pid=stage['worker_pid']
            assert 'size_aware_study.pipeline' in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
            assert 'size_aware_study.train' in Path(f'/proc/{worker_pid}/cmdline').read_bytes().decode()
            os.kill(pid,signal.SIGSTOP)
            confirmed=read(root/'stage.json')
            if confirmed!=stage or (root/'detail').exists():os.kill(pid,signal.SIGCONT);status(status='not_activated',reason='original controller advanced during preflight');return
            old_controller=pid;adopted=True
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            cmd=[sys.executable,'-u','-m','benchmarks.v5.size_aware_study.train',*args('detail')]
            log=(out/'detail_formal.log').open('x');detail=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt=status(status='running_parallel',stage='precision_and_detail_formal',preserved_precision_worker_pid=worker_pid,
                           detail_worker_pid=detail.pid,detail_GPU=gpu,precision_GPU=3,free_before_launch_MiB=free[gpu],command=cmd,
                           suspended_original_controller_pid=pid,execution_contract_sha256=sha(root/'contract.json'))
            canonical(receipt);(out/'handoff_receipt.json').write_text(json.dumps(receipt,indent=2));break
        precision_done=False;detail_done=False
        while not (precision_done and detail_done):
            if not precision_done:
                state,exit_code=process_state(worker_pid)
                if state is None:raise RuntimeError('Preserved worker disappeared without capturable exit status')
                if state=='Z':
                    assert os.waitstatus_to_exitcode(exit_code)==0,('precision_worker_exit',exit_code)
                    assert read(root/'precision/status.json')['stage']=='complete_unaccepted'
                    call('precision_accept','benchmarks.v5.size_aware_study.train',args('precision')+['--accept'])
                    precision_done=True
            if not detail_done and detail.poll() is not None:
                rc=detail.returncode;log.write('\nEXIT_CODE='+str(rc)+'\n');log.flush();assert rc==0,('detail_worker_exit',rc)
                call('detail_accept','benchmarks.v5.size_aware_study.train',args('detail')+['--accept']);detail_done=True
            status(status='running_parallel',stage='formal_or_acceptance',precision_accepted=precision_done,detail_accepted=detail_done,
                   preserved_precision_worker_pid=worker_pid,detail_worker_pid=detail.pid,detail_GPU=gpu)
            if not (precision_done and detail_done):time.sleep(10)
        for arm in ['control','precision','detail']:
            assert read(root/arm/'independent_acceptance.json')['status']=='accepted_P3_spatial_evaluation'
        canonical(status(status='complete_runs',P3_complete=False))
        call('comparison','benchmarks.v5.size_aware_study.compare',['--root',str(root)])
        canonical(status(status='complete_accepted',P3_complete=False,parallel_resource_change_only=True,next_action='report; no automatic next experiment'))
    except Exception as error:
        value=status(status='failed',error=repr(error),retry=False)
        if adopted:canonical(value)
        raise
    finally:
        if old_controller is not None:
            # Only the suspended orchestration process is retired; training workers are never killed here.
            try:os.kill(old_controller,signal.SIGTERM);os.kill(old_controller,signal.SIGCONT)
            except ProcessLookupError:pass


if __name__=='__main__':main()
