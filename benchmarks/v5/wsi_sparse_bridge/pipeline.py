import argparse,json,subprocess,os,sys
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));jobs=[];records=[]
 def start(module,args,gpu,log):
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=12288,(gpu,free)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VIPS_CONCURRENCY='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2');cmd=[sys.executable,'-m','benchmarks.v5.wsi_sparse_bridge.'+module]+args;f=log.open('x');proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);entry=dict(pid=proc.pid,gpu=gpu,module=module,cmd=cmd,log=str(log),status='running');records.append(entry);dump(r/'launch.json',records);return proc,f,entry
 def wait(job):
  p,f,e=job;rc=p.wait();f.write(f'\nEXIT_CODE={rc}\n');f.close();e.update(status='complete' if rc==0 else 'failed',exit_code=rc);dump(r/'launch.json',records);return rc
 integration=start('integration',['--root',str(r)],1,r/'integration.log');assert wait(integration)==0
 for arm,gpu in [('wsi_sam_ft',1)]:
  (r/arm).mkdir();jobs.append((arm,start('workflow',['--root',str(r),'--arm',arm],gpu,r/arm/'workflow.log')))
 dump(r/'stage.json',dict(status='single_sparse_bridge_workflow_running',pid=os.getpid()));successful=[]
 for arm,j in jobs:
  if wait(j)==0:successful.append(arm)
 evaljobs=[]
 for arm in successful:
  subprocess.run([sys.executable,'-m','benchmarks.v5.wsi_sparse_bridge.prepare_eval','--root',str(r),'--arm',arm],check=True);er=r/arm/'evaluation';gpu=1;gate=start('evaluate',['--root',str(er),'--method',arm,'--shard','0','--gate'],gpu,er/'gate.log')
  if wait(gate)!=0:continue
  for shard,gpu in enumerate([1,5]):evaljobs.append((arm,start('evaluate',['--root',str(er),'--method',arm,'--shard',str(shard)],gpu,er/f'shard{shard}.log')))
 results={arm:[] for arm in successful}
 for arm,j in evaljobs:results[arm].append(wait(j))
 accepted=[]
 for arm,codes in results.items():
  if len(codes)==2 and not any(codes):
   subprocess.run([sys.executable,'-m','benchmarks.v5.wsi_sparse_bridge.aggregate','--root',str(r/arm/'evaluation')],check=True);accepted.append(arm)
 dump(r/'stage.json',dict(status='complete_accepted' if len(accepted)==1 else 'partial_or_failed_stop',accepted_arms=accepted,pid=os.getpid()))
if __name__=='__main__':main()
