"""Gate first, two disjoint inference shards on two GPUs, then aggregate."""
import argparse,subprocess,sys,os,json,time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();c=json.load(open(a.root/'contract.json'));out=a.root/a.arm
 def launch(stage,module,args,gpu):
  for f,h in c['sources'].items():assert sha(f)==h,f
  while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<12288:
   dump(out/(stage+'_resource_wait.json'),dict(status='waiting_free_memory',gpu=gpu,at=time.time()));time.sleep(30)
  cmd=[sys.executable,'-u','-m','benchmarks.v5.hg_paper_ablation.'+module,*map(str,args)];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));log=(out/(stage+'.log')).open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  item=dict(stage=stage,pid=proc.pid,gpu=gpu,command=cmd);dump(out/(stage+'_worker.json'),item);return proc,log,item
 def finish(job):
  proc,log,item=job;rc=proc.wait();log.write(f'\nEXIT_CODE={rc}\n');log.close();return rc
 def run(stage,module,args):
  job=launch(stage,module,args,int(os.environ['CUDA_VISIBLE_DEVICES']));dump(out/'evaluation_stage.json',dict(status='running',**job[2]));assert finish(job)==0,stage
 try:
  run('prepare_eval','prepare_eval',['--root',a.root,'--arm',a.arm]);er=out/'evaluation'
  run('gate_eval','evaluate',['--root',er,'--method',a.arm,'--shard',0,'--gate'])
  jobs=[]
  for shard,gpu in enumerate(c['eval_gpus'][a.arm]):
   jobs.append(launch('eval_'+str(shard),'evaluate',['--root',er,'--method',a.arm,'--shard',shard],gpu));dump(out/'evaluation_stage.json',dict(status='parallel_shards_running',workers=[j[2] for j in jobs]))
  codes=[finish(j) for j in jobs];assert codes==[0,0],codes
  run('aggregate','aggregate',['--root',er]);dump(out/'evaluation_stage.json',dict(status='complete_accepted'))
 except Exception as e:dump(out/'evaluation_stage.json',dict(status='failed',error=repr(e)));raise
if __name__=='__main__':main()
