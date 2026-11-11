import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_erosion
from benchmarks.v5.region_evidence_study.model import DecoderData
from benchmarks.v5.wsi_quality_study.prepare import dump
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));plans=json.load(open(r/'tasks.json'));data=DecoderData(Path(c['cache']),'match');rows=[]
for arm in ['original','sansa']:
 packed=np.load(Path(c['baseline_diagnostic'])/f'{arm}_masks.npy',mmap_mode='r')
 for i,t in enumerate(plans):
  gt,mp=data.labels(t['query_id']);truth=gt==t['target'];valid=gt!=255;pred=np.unpackbits(packed[2*i+1],count=512*512).reshape(512,512).astype(bool);edge=(truth^binary_erosion(truth,border_value=0))&valid
  if not edge.any():continue
  dist=distance_transform_edt(~edge,sampling=(mp[1],mp[0]));fp=pred&~truth&valid;fn=~pred&truth&valid;z=dict(arm=arm,index=i,task=t,fp=int(fp.sum()),fn=int(fn.sum()))
  for band in [2,8]:z[f'fp_farther_{band}um']=int((fp&(dist>band)).sum());z[f'fn_farther_{band}um']=int((fn&(dist>band)).sum())
  rows.append(z)
dump(r/'boundary_report.json',dict(status='complete_saved_mask_diagnostic',rows=rows,limitation='Distances from positive GT boundary; no target/edge cases excluded; not an attainable improvement or deployable oracle. Fixed train tasks.'))
