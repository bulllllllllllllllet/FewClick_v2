"""Prepare once, launch exactly two named tmux arms, preserve terminal logs."""
import argparse,subprocess,sys,os,json,shlex,time
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 # No duplicated study worker, regardless of current GPU utilization.
 active=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
 assert not any('benchmarks.v5.hg_paper_ablation.run --root' in line for line in active)
 old=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/sansa_cache_1113_004500/queue_1113_005000/jobs.json')
 jobs=json.load(open(old));assert all(j['status']=='complete' for j in jobs)
 for gpu in [3,4]:assert int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=12288
 subprocess.run([sys.executable,'-m','benchmarks.v5.hg_paper_ablation.prepare','--root',str(a.root)],check=True)
 c=json.load(open(a.root/'contract.json'));(a.root/'tmp').mkdir();receipt=[]
 for arm,gpu in c['gpus'].items():
  log=a.root/arm/'launcher.log';session='paper_ablation_'+arm+'_'+a.root.name.rsplit('_',2)[-2]+'_'+a.root.name.rsplit('_',1)[-1]
  cmd=[sys.executable,'-u','-m','benchmarks.v5.hg_paper_ablation.run','--root',str(a.root),'--arm',arm]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=str(a.root/'tmp'),PYTHONPATH='/nfs-medical3/zyh/fewclick_v2/vendor/phikon-v2:'+str(Path.cwd()),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VIPS_CONCURRENCY='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',VIPS_DISC_THRESHOLD='100m')
  shell=shlex.join(['env', *[k+'='+env[k] for k in ['CUDA_VISIBLE_DEVICES','TMPDIR','PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS','VIPS_CONCURRENCY','CUBLAS_WORKSPACE_CONFIG','OPENBLAS_NUM_THREADS','VIPS_DISC_THRESHOLD']], *cmd])+' > '+shlex.quote(str(log))+' 2>&1; rc=$?; echo EXIT_CODE=$rc >> '+shlex.quote(str(log))+'; exit $rc'
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(Path.cwd()),shell],env=env,check=True)
  receipt.append(dict(arm=arm,gpu=gpu,session=session,command=cmd,log=str(log)))
 dump(a.root/'launch.json',dict(status='launched_not_experiment_complete',workers=receipt))
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
