import argparse,subprocess,os,sys,json
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from .prepare import ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--gpus',nargs=2,type=int,required=True);a=p.parse_args();r=ROOT/a.method;procs=[];records=[]
 def start(shard,gpu,gate=False):
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=12288
  cmd=[sys.executable,'-m','benchmarks.v5.additional_sam.worker','--root',str(r),'--method',a.method,'--shard',str(shard)]+(['--gate'] if gate else []);log=r/('gate.log' if gate else f'shard{shard}.log');f=log.open('x');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VIPS_CONCURRENCY='2',CUBLAS_WORKSPACE_CONFIG=':4096:8');proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);entry=dict(pid=proc.pid,method=a.method,gpu=gpu,shard=shard,gate=gate,cmd=cmd,log=str(log),status='running');records.append(entry);dump(r/'launch.json',records);return proc,f,entry
 def wait(job):
  p,f,e=job;rc=p.wait();f.write(f'\nEXIT_CODE={rc}\n');f.close();e.update(exit_code=rc,status='complete' if rc==0 else 'failed');dump(r/'launch.json',records);return rc
 dump(r/'stage.json',dict(status='gate_running',pid=os.getpid()));assert wait(start(0,a.gpus[0],True))==0
 dump(r/'stage.json',dict(status='two_shards_running',pid=os.getpid()));jobs=[start(i,g) for i,g in enumerate(a.gpus)];assert all(wait(job)==0 for job in jobs)
 dump(r/'stage.json',dict(status='all11000_masks_independently_accepted_aggregation_pending',pid=os.getpid()))
if __name__=='__main__':main()
