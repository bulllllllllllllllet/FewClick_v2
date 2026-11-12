import argparse,json,os,subprocess,sys
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));arm=r/a.arm
 stages=[('functional',['--cache',c['cache'],'--output',str(arm/'functional'),'--arm',a.arm,'--protocol',c['functional_source']]),('smoke',['--root',str(r),'--arm',a.arm,'--smoke']),('formal',['--root',str(r),'--arm',a.arm])]
 for name,args in stages:
  module='functional' if name=='functional' else 'train';cmd=[sys.executable,'-m','benchmarks.v5.sam_current_fair.'+module]+args
  with (arm/f'{name}.log').open('x') as f:
   proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);dump(arm/'stage.json',dict(status=name+'_running',worker_pid=proc.pid,controller_pid=os.getpid(),cmd=cmd));rc=proc.wait();f.write(f'\nEXIT_CODE={rc}\n')
  if rc:dump(arm/'stage.json',dict(status=name+'_failed',exit_code=rc,worker_pid=proc.pid));raise RuntimeError(name+' failed; do not continue')
 dump(arm/'stage.json',dict(status='training_accepted_evaluation_pending',controller_pid=os.getpid()))
if __name__=='__main__':main()
