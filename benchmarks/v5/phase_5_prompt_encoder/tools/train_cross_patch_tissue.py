#!/usr/bin/env python3
"""Train prompt heads on spatially separated support/query episodes."""
import argparse, hashlib, json, os, re, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from benchmarks.v4.phase_5_prompt_encoder.src.dataset import PromptEpisodeDataset, collate_prompt_episodes
from benchmarks.v5.phase_5_prompt_encoder.src.learnable_tissue_prototype import LearnableTissuePrototype, unprompted_bce
from benchmarks.v4.phase_5_prompt_encoder.src.model import PromptRegionModel
from benchmarks.v5.phase_5_prompt_encoder.src.cross_patch_support import separated_contexts, transfer_support

SPLIT=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235')

def model(name): return PromptRegionModel() if name=='phase5' else LearnableTissuePrototype(name)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--timestamp',required=True); p.add_argument('--steps',type=int,default=2); p.add_argument('--per-stratum',type=int,default=1); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--device',default='cuda:1'); p.add_argument('--seed',type=int,default=17); a=p.parse_args()
 if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp): raise ValueError('bad timestamp')
 out=Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt')/('cross_patch_train_'+a.timestamp); out.mkdir(parents=True,exist_ok=False)
 d=PromptEpisodeDataset(SPLIT/'cache_index.parquet',SPLIT/'labels_index.parquet',SPLIT/'patches_index.parquet','train',20260922,{'point':.4,'small':.35,'large':.25},tuple(range(12)),eligibility_index=SPLIT/'eligibility_index.parquet')
 spatial=__import__('pandas').read_parquet(SPLIT/'spatial_manifest.parquet').set_index('patch_id')
 def box(pid):
  r=spatial.loc[pid]; return tuple(float(r[k]) for k in ('context_x0','context_y0','context_x1','context_y1'))
 rng=np.random.default_rng(20260922); pairs=[]
 for c in range(12):
  for z in ('point','small','large'):
   qs=np.flatnonzero((d.target_classes==c)&(d.prompt_sizes==z)); rng.shuffle(qs)
   for qi in qs:
    cand=np.flatnonzero((d.wsi_ids==d.wsi_ids[qi])&(d.target_classes==c)&(d.prompt_sizes==z)); rng.shuffle(cand)
    si=next((int(x) for x in cand if int(x)!=int(qi) and separated_contexts(box(d.patch_ids[qi]),box(d.patch_ids[x]))),None)
    if si is not None: pairs.append((int(qi),si)); break
   if len([x for x in pairs if d.target_classes[x[0]]==c and d.prompt_sizes[x[0]]==z])<a.per_stratum: raise ValueError('no pair')
   # retain requested count deterministically by continuing if needed
   while len([x for x in pairs if d.target_classes[x[0]]==c and d.prompt_sizes[x[0]]==z])<a.per_stratum: raise ValueError('insufficient pair')
 pairs=pairs[:12*3*a.per_stratum]
 items=[(d[q],d[s]) for q,s in pairs]; order=np.resize(np.arange(len(items)),a.steps*a.batch_size)
 def batch(xs): return {k:v.to(a.device) if torch.is_tensor(v) else v for k,v in collate_prompt_episodes(xs).items()}
 torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); records=[]
 for name in ('phase5','single','multi'):
  m=model(name).to(a.device).train(); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
  for step in range(a.steps):
   xs=[]
   for j in order[step*a.batch_size:(step+1)*a.batch_size]:
    q,s=items[int(j)]; xs.append(transfer_support(q,s))
   b=batch(xs); opt.zero_grad(set_to_none=True); loss=unprompted_bce(m(b)['logits'],b); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True); opt.step(); records.append(dict(method=name,step=step+1,loss=float(loss)))
  torch.save(dict(model=m.cpu().state_dict(),method=name,seed=a.seed,steps=a.steps,data_hash=hashlib.sha256(json.dumps(pairs).encode()).hexdigest()),out/f'{name}_seed{a.seed}_step{a.steps}.pth')
 (out/'manifest.json').write_text(json.dumps(dict(protocol='cross patch support/query',pairs=[dict(query=d.patch_ids[q],support=d.patch_ids[s]) for q,s in pairs],records=records),indent=2)); print(json.dumps(dict(output=str(out),pairs=len(pairs),records=len(records))),flush=True)
if __name__=='__main__': main()
