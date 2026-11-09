import argparse,json,os,subprocess,sys
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'))
 def run(stage,module,args,gpu):
  for f,h in c['sources'].items():assert sha(f)==h
  free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=12288
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));cmd=[sys.executable,'-u','-m','benchmarks.v5.sam2_encoder_probe.'+module,*map(str,args)]
  with (r/(stage+'.log')).open('x') as f:
   proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);dump(r/'stage.json',dict(status='running',stage=stage,worker_pid=proc.pid,controller_pid=os.getpid(),command=cmd));rc=proc.wait();f.write(f'\nEXIT_CODE={rc}\n');assert rc==0,(stage,rc)
 try:
  run('workflow','workflow',['--root',r,'--arm','sam2_encoder'],3)
  run('prepare_eval','prepare_eval',['--root',r,'--arm','sam2_encoder'],3)
  er=r/'sam2_encoder/evaluation';run('eval_gate','evaluate',['--root',er,'--method','sam2_encoder','--shard',0,'--gate'],3)
  # Serial two shards, preserving active WSI workers and limiting resource footprint.
  for shard in [0,1]:run('eval_'+str(shard),'evaluate',['--root',er,'--method','sam2_encoder','--shard',shard],3)
  run('aggregate','aggregate',['--root',er],3)
  dump(r/'stage.json',dict(status='complete_accepted'))
 except Exception as e:dump(r/'stage.json',dict(status='failed',error=repr(e),no_automatic_retry=True));raise
if __name__=='__main__':main()
