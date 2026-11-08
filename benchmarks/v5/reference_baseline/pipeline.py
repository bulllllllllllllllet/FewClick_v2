"""Wait for official weights, run one bounded readonly pilot, preserve failures."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);r=p.parse_args().root
 def record(**kw):(r/'stage.json').write_text(json.dumps(dict(controller_pid=os.getpid(),**kw),indent=2)+'\n')
 record(status='preparing_waiting_official_weights')
 start=time.monotonic()
 while not (r/'weights.json').exists():
  log=(r/'prepare.log').read_text() if (r/'prepare.log').exists() else ''
  if 'EXIT_CODE=' in log:record(status='preparation_failed_stop',log=str(r/'prepare.log'));raise RuntimeError('official weight preparation failed')
  if time.monotonic()-start>1800:record(status='preparation_timeout_stop');raise RuntimeError('weight preparation timeout')
  time.sleep(5)
 gpu=3;free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=12288,(gpu,free)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VIPS_CONCURRENCY='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2')
 cmd=[sys.executable,'-u','-m','benchmarks.v5.reference_baseline.pilot','--root',str(r)]
 with (r/'pilot.log').open('x') as f:
  worker=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);record(status='pilot_running',worker_pid=worker.pid,gpu=gpu,cmd=cmd,log=str(r/'pilot.log'));rc=worker.wait();f.write(f'\nEXIT_CODE={rc}\n')
 record(status='complete_accepted' if rc==0 and (r/'acceptance.json').exists() else 'pilot_failed_stop',worker_pid=worker.pid,exit_code=rc,gpu=gpu)
 if rc:raise RuntimeError('pilot failed, preserve without replacement')
if __name__=='__main__':main()
