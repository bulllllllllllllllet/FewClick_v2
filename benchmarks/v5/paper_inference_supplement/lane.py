"""Sequential jobs on one GPU; fail-stop, real child PID and exit receipts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from benchmarks.v5.wsi_quality_study.prepare import dump
LANES={'quality0':[('local','HG_mix'),('robust','HG_mix'),('local','sam')],
       'quality1':[('local','sansa'),('robust','sansa'),('local','sam2')],
       'quality4':[('local','wsi'),('local','med2d')],
       'timing3':[('timing',m) for m in ['HG_mix','sansa','sam','sam2','wsi','med2d']]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--lane',choices=LANES,required=True);a=p.parse_args()
    out=a.root/a.lane;out.mkdir();completed=[]
    for stage,method in LANES[a.lane]:
        cmd=[sys.executable,'-u','-m','benchmarks.v5.paper_inference_supplement.worker','--root',str(a.root),'--stage',stage,'--method',method]
        with (out/f'{stage}_{method}.log').open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            dump(out/'stage.json',dict(status='running',stage=stage,method=method,pid=proc.pid,controller_pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=cmd,completed=completed))
            rc=proc.wait();log.write(f'\nEXIT_CODE={rc}\n')
        if rc:
            dump(out/'stage.json',dict(status='failed',stage=stage,method=method,pid=proc.pid,exit_code=rc,completed=completed));raise RuntimeError(f'{stage}/{method} failed; queue stopped')
        receipt=json.loads((a.root/f'{stage}_{method}'/'acceptance.json').read_text());assert receipt['status']=='complete_accepted'
        completed.append([stage,method])
    dump(out/'stage.json',dict(status='complete_accepted',completed=completed,controller_pid=os.getpid()))
if __name__=='__main__':main()
