"""Fixed train-only support/query diagnosis; no model or prompt selection."""
import argparse,json,os
from pathlib import Path
import numpy as np,torch
from benchmarks.v5.same_slide_retrain.model import build as baseline
from .model import DecoderData
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2)
 out=r/'diagnostic';out.mkdir();plans=json.load(open(r/'diagnostic_tasks.json'));data=DecoderData(Path(c['cache']),'match');allrows=[]
 for arm in ['original','sansa']:
  m=baseline(arm,c['diagnostic_checkpoints'][arm]);m.eval();rows=[];masks=[]
  with torch.no_grad():
   for i,t in enumerate(plans):
    assert data.rows[t['support_id']]['split']==data.rows[t['query_id']]['split']=='train'
    z=m(**data.model_inputs(t,'xcell','cuda:0'));row=dict(task=t)
    for side in ['support','query']:
     gt,mpp=data.labels(t[side+'_id']);pred=z[side]['pixel_probability'][0].cpu().numpy()>=.5
     row[side]=score(pred,gt==t['target'],gt!=255,mpp);masks.append(np.packbits(pred))
    rows.append(row)
    if i%8==0:dump(out/'progress.json',dict(pid=os.getpid(),arm=arm,done=i+1,total=len(plans)))
  np.save(out/f'{arm}_masks.npy',np.stack(masks));dump(out/f'{arm}_rows.json',rows);del m;torch.cuda.empty_cache()
 dump(out/'acceptance.json',dict(status='complete_train_only_diagnostic',tasks=len(plans),selection_used=False,threshold=.5))
if __name__=='__main__':main()
