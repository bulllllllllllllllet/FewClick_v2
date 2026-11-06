"""Queue gated seed replication, fixed endpoints and independent evaluation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));arm=c['arm'];assert not (r/'history.jsonl').exists()
 def status(**kw):
  v=dict(controller_pid=os.getpid(),timestamp=time.time(),**kw);(r/'stage.json').write_text(json.dumps(v));
  with (r/'history.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 def run(label,module,args,gpu=True):
  for path,h in c['sources'].items():assert sha(path)==h,path
  if gpu:
   while int(subprocess.check_output(['nvidia-smi','--id='+str(c['GPU']),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12288:status(status='waiting_GPU_memory',stage=label);time.sleep(10)
  cmd=[sys.executable,'-u','-m',module,*map(str,args)]
  with (r/(label+'.log')).open('w') as log:
   worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);status(status='running',stage=label,worker_pid=worker.pid,command=cmd);rc=worker.wait();log.write('\nEXIT_CODE='+str(rc)+'\n');assert rc==0,(label,rc)
 try:
  for path,h in c['weights'].items():assert sha(path)==h
  run('functional','benchmarks.v5.paper_completion.functional',['--cache',c['cache'],'--output',r/'learning'/arm,'--arm',arm,'--seed',c['seed']])
  run('functional_accept','benchmarks.v5.architecture_bridge.accept_spatial_learning',['--cache',c['cache'],'--run',r/'learning'/arm],False)
  common=['--arm',arm,'--cache',c['cache'],'--protocol',c['protocol'],'--learning',r/'learning','--seed',c['seed'],'--execution-contract',r/'contract.json']
  for smoke in [True,False]:
   args=common+['--output',r/('smoke' if smoke else 'control')]+(['--smoke'] if smoke else [])
   run('smoke' if smoke else 'formal','benchmarks.v5.paper_completion.train',args)
   run('smoke_accept' if smoke else 'formal_accept','benchmarks.v5.paper_completion.train',args+['--accept'],False)
  status(status='complete_accepted',next_action='report, no automatic next study')
 except Exception as exc:status(status='failed',error=repr(exc),retry=False);raise
if __name__=='__main__':main()
