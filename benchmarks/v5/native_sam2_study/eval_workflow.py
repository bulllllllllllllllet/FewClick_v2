import argparse,subprocess,sys,os,json
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();c=json.load(open(a.root/'contract.json'));out=a.root/a.arm
 def run(stage,module,args):
  for f,h in c['sources'].items():assert sha(f)==h,f
  gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=12288
  cmd=[sys.executable,'-u','-m','benchmarks.v5.native_sam2_study.'+module,*map(str,args)]
  with (out/(stage+'.log')).open('x') as f:
   child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);dump(out/'evaluation_stage.json',dict(status='running',stage=stage,pid=child.pid,gpu=gpu,command=cmd));rc=child.wait();f.write(f'\nEXIT_CODE={rc}\n');assert rc==0,(stage,rc)
 try:
  run('prepare_eval','prepare_eval',['--root',a.root,'--arm',a.arm]);er=out/'evaluation'
  run('gate_eval','evaluate',['--root',er,'--method',a.arm,'--shard',0,'--gate'])
  for shard in [0,1]:run('eval_'+str(shard),'evaluate',['--root',er,'--method',a.arm,'--shard',shard])
  run('aggregate','aggregate',['--root',er]);dump(out/'evaluation_stage.json',dict(status='complete_accepted'))
 except Exception as e:dump(out/'evaluation_stage.json',dict(status='failed',error=repr(e)));raise
if __name__=='__main__':main()
