import argparse,json,subprocess,sys,os,time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));active={};records=[];done=set();failed=False;detail_started=False
 for f,h in c['weights'].items():assert sha(f)==h,f
 for field in ['training_consumption','functional_source','monitor_source']:assert sha(c[field])==c[field+'_sha256']
 def start(arm,stage,gpu):
  for f,h in c['sources'].items():assert sha(f)==h,f
  assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=12288
  module='workflow' if stage=='train' else 'eval_workflow';cmd=[sys.executable,'-u','-m','benchmarks.v5.same_slide_retrain.'+module,'--root',str(r),'--arm',arm];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));handle=(r/arm/(stage+'_controller.log')).open('x');proc=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT);entry=dict(arm=arm,stage=stage,gpu=gpu,pid=proc.pid,command=cmd,status='running');records.append(entry);active[arm,stage]=(proc,handle,entry);dump(r/'launch.json',records)
 start('original','train',3);start('sansa','train',1);start('detail','train',4);detail_started=True
 while active:
  finished=[]
  for key,(proc,handle,entry) in list(active.items()):
   rc=proc.poll()
   if rc is None:continue
   handle.write(f'\nEXIT_CODE={rc}\n');handle.close();entry.update(status='complete' if rc==0 else 'failed',exit_code=rc);finished.append((key,rc));del active[key]
   if rc:failed=True
   else:done.add(key)
  for (arm,stage),rc in finished:
   if rc==0 and stage=='train' and not failed:
    start(arm,'eval',3 if arm=='original' else 1 if arm=='sansa' else 4)
  dump(r/'launch.json',records);dump(r/'stage.json',dict(status='failed_waiting_active_workers' if failed else 'running',controller_pid=os.getpid(),complete=[list(k) for k in sorted(done)],active=[list(k) for k in active],detail_started=detail_started))
  if active:time.sleep(5)
 dump(r/'stage.json',dict(status='complete_accepted' if not failed and len(done)==6 else 'failed_stop_no_retry',complete=[list(k) for k in sorted(done)],controller_pid=os.getpid()))
if __name__=='__main__':main()
