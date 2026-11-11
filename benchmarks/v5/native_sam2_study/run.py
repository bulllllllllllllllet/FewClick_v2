import argparse,json,subprocess,sys,os
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));out=r/a.arm
 try:
  for stage,module in [('engineering','engineering'),('training','workflow'),('evaluation','eval_workflow')]:
   for f,h in c['sources'].items():assert sha(f)==h,f
   for k in ['training_consumption','functional_source','monitor_source']:assert sha(c[k])==c[k+'_sha256']
   for f,h in c['weights'].items():assert sha(f)==h,f
   gpu=os.environ['CUDA_VISIBLE_DEVICES'];assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=12288
   cmd=[sys.executable,'-u','-m','benchmarks.v5.native_sam2_study.'+module,'--root',str(r),'--arm',a.arm]
   with (out/(stage+'_controller.log')).open('x') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);dump(out/'controller_stage.json',dict(status=stage+'_running',worker_pid=proc.pid,controller_pid=os.getpid(),command=cmd));rc=proc.wait();log.write(f'\nEXIT_CODE={rc}\n')
   assert rc==0,(stage,rc)
  dump(out/'controller_stage.json',dict(status='complete_accepted',quality_promotion=False))
 except Exception as e:dump(out/'controller_stage.json',dict(status='failed_stop_no_retry',error=repr(e)));raise
if __name__=='__main__':main()
