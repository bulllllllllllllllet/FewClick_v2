"""Persistent coordinator: real gates then six nonduplicated read-only shards."""
import argparse,os,subprocess,time,sys,json,traceback
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    assert (r/'contract.json').is_file();launch=[]
    def start(method,shard,gpu,gate=False):
        free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=12288,(gpu,free)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VIPS_CONCURRENCY='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2')
        cmd=[sys.executable,'-m','benchmarks.v5.wsi_same_slide.worker','--root',str(r),'--method',method,'--shard',str(shard)]+(['--gate'] if gate else [])
        log=r/(f'gate_{method}.log' if gate else f'{method}_{shard}.log');f=log.open('x');proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        entry=dict(pid=proc.pid,method=method,shard=shard,gpu=gpu,gate=gate,cmd=cmd,log=str(log),status='running');launch.append(entry);dump(r/'launch.json',launch);return proc,f,entry
    def finish(group):
        remaining=list(group)
        while remaining:
            for job in remaining[:]:
                proc,f,entry=job;rc=proc.poll()
                if rc is None:continue
                f.write(f'\nEXIT_CODE={rc}\n');f.close();entry.update(status='complete' if rc==0 else 'failed',exit_code=rc);dump(r/'launch.json',launch);remaining.remove(job)
            if remaining:time.sleep(5)
        assert all(j[2]['exit_code']==0 for j in group),'worker failed; inspect retained logs'
    try:
        dump(r/'stage.json',dict(status='real_forward_gates_running',controller_pid=os.getpid()))
        finish([start('full',0,1,True),start('no_match',0,3,True),start('sam',0,5,True)])
        dump(r/'stage.json',dict(status='six_shard_inference_running',controller_pid=os.getpid()))
        finish([start(method,shard,2*mi+shard) for mi,method in enumerate(['full','no_match','sam']) for shard in [0,1]])
        dump(r/'stage.json',dict(status='independent_acceptance_running',controller_pid=os.getpid()))
        with (r/'collect.log').open('x') as f:
            rc=subprocess.call([sys.executable,'-m','benchmarks.v5.wsi_same_slide.collect','--root',str(r)],stdout=f,stderr=subprocess.STDOUT);f.write(f'\nEXIT_CODE={rc}\n');assert rc==0
        dump(r/'stage.json',dict(status='complete_accepted',controller_pid=os.getpid(),acceptance=str(r/'acceptance.json')))
    except Exception:
        dump(r/'stage.json',dict(status='failed',controller_pid=os.getpid(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
