"""Read-only cell fusion probes; perturbations are diagnostics, not candidate scores."""
import argparse,json,copy
from pathlib import Path
import numpy as np
import torch
from .model import DecoderModel,DecoderData
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;torch.set_num_threads(2)
 base=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');data=DecoderData(base/'cell_repair_rebalance_1102_210552/normalized','cell_on');m=DecoderModel('cell_on');ck=torch.load(base/'matcher_cell_gatefix_1106_195114/control/checkpoint.pt',weights_only=True,map_location='cuda:0');m.load_state_dict(ck['model']);m.eval();before=tree_sha(m.state_dict())
 plans=list(map(json.loads,(base/'phikon_joint_tasks_1031_222610/data/training4000.jsonl').read_text().splitlines()));chosen=[]
 for d in ['crc','tiger']:
  for k in ['local','cross_present','cross_absent']:chosen.append(next(x for x in plans if x['dataset']==d and x['kind']==k))
 evidence=[];current=[]
 def hook(module,args,out):
  tokens,mass,cells,valid,total=args
  if 'attention' not in out:return
  w=out['attention'];n=cells.shape[1];counts=out['region_cell_count'];ctx=out['cell_context'];s=valid.sum().item()
  current.append(dict(n=n,total=float(total[0]),expansion=float(total[0])/max(1,s),xy_min=float(cells[...,:2].min()) if n else None,xy_max=float(cells[...,:2].max()) if n else None,embedding_std=float(cells[...,10:].std()) if n else None,mean_region_count=float(counts.mean()),min_region_count=float(counts.min()),context_token_norm_ratio=float(ctx.norm(dim=-1).mean()/tokens.norm(dim=-1).mean().clamp_min(1e-8)),effective_cells=float((1/w.square().sum(-1).clamp_min(1e-8)).mean()) if n else 0,mass_sum_error=float((mass.sum(1)-1).abs().max()) if n else 0))
 handle=m.base.core.cell.register_forward_hook(hook)
 with torch.no_grad():
  for i,plan in enumerate(chosen):
   original=data.model_inputs(plan,'xcell','cuda:0');gt,mpp=data.labels(plan['query_id']);variants={}
   for variant in ['normal','zero_embedding','selected_count','no_cell_fusion']:
    x=copy.deepcopy(original);current.clear();m.base.core.cell.arm='I' if variant=='no_cell_fusion' else 'S11'
    for side in ['support','query']:
     for suffix in ['','_1','_2']:
      if variant=='zero_embedding':x[side]['cells'+suffix][...,10:]=0
      if variant=='selected_count':x[side]['total_cell_count' if not suffix else 'total'+suffix]=x[side]['cell_valid'+suffix].sum(1).float()
    prob=m(**x)['query']['pixel_probability'][0].cpu().numpy();variants[variant]=dict(metrics=score(prob>=.5,gt==plan['target'],gt!=255,mpp),mean_probability=float(prob.mean()),fusion=copy.deepcopy(current))
   evidence.append(dict(dataset=plan['dataset'],kind=plan['kind'],target=plan['target'],variants=variants));print('completed',i+1,flush=True)
 handle.remove();assert tree_sha(m.state_dict())==before
 mapper=ck['model']['base.representation.mapper.weight'];identity=torch.eye(64,device=mapper.device)
 (r/'report.json').write_text(json.dumps(dict(status='complete_readonly_diagnostic',rows=evidence,mapper_max_identity_change=float((mapper-identity).abs().max()),model_state_unchanged=True,limitation='six fixed training tasks; inference interventions are distribution shifts, not retrained ablations; no causal or generalization acceptance'),indent=2))
if __name__=='__main__':main()
