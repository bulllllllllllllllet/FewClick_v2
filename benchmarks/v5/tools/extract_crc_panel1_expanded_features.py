#!/usr/bin/env python3
"""Build aligned 128/256 features for the fully expanded Panel1 manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from benchmarks.v5.tools.probe_crc_panel1_multiscale_context import context256
from benchmarks.v5.tools.run_crc_panel1_cell_identity import PREPROCESS, SOURCE, WEIGHTS, load_ctranspath
from benchmarks.v5.tools.cache_crc_ctranspath_cell_features import crop_mask,crop_region,soft_attention

def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--base',type=Path,required=True); p.add_argument('--base128',type=Path,required=True); p.add_argument('--base256',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:1'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(8)
 data=pd.read_csv(a.manifest/'cells.csv'); meta=json.loads((a.manifest/'manifest.json').read_text()); old=pd.read_csv(a.base/'cells.csv');
 with np.load(a.base128/'features.npz') as z: old128=z['features']
 with np.load(a.base256/'features.npz') as z: old256=z['features']
 lookup={(r.patch,int(r.he_id)):(old128[i],old256[i]) for i,r in enumerate(old.itertuples())}
 x128=np.full((len(data),768),np.nan,np.float32); x256=x128.copy(); missing=[]
 for i,r in enumerate(data.itertuples()):
  v=lookup.get((r.patch,int(r.he_id)))
  if v is None: missing.append(i)
  else: x128[i],x256[i]=v
 encoder=load_ctranspath(WEIGHTS,torch.device(a.device)).requires_grad_(False)
 for patch,frame in data.iloc[missing].groupby('patch',sort=False):
  image=np.asarray(Image.open(SOURCE/'registered_he_patches'/f'{patch}.png').convert('RGB')); mask=np.load(SOURCE/'cell_matching/masks'/f'{patch}.npy',allow_pickle=False)
  t128=[PREPROCESS(soft_attention(crop_region(image,round(r.he_y),round(r.he_x)),crop_mask(mask,round(r.he_y),round(r.he_x))==int(r.he_id))) for r in frame.itertuples()]
  t256=[PREPROCESS(context256(image,mask,r)) for r in frame.itertuples()]
  with torch.inference_mode():
   v128=torch.cat([encoder(torch.stack(t128[j:j+32]).to(a.device)).cpu() for j in range(0,len(t128),32)]).numpy(); v256=torch.cat([encoder(torch.stack(t256[j:j+32]).to(a.device)).cpu() for j in range(0,len(t256),32)]).numpy()
  x128[frame.index]=v128; x256[frame.index]=v256
 if not np.isfinite(x128).all() or not np.isfinite(x256).all(): raise ValueError('nonfinite or incomplete features')
 np.savez_compressed(a.output/'features.npz',features128=x128,features256=x256); (a.output/'feature_audit.json').write_text(json.dumps({'status':'complete','manifest_sha256':meta['manifest_sha256'],'cells':len(data),'reused':len(data)-len(missing),'extracted':len(missing),'feature_dim':768},indent=2)); print(json.dumps({'status':'complete','cells':len(data),'reused':len(data)-len(missing),'extracted':len(missing)}),flush=True)
if __name__=='__main__': main()
