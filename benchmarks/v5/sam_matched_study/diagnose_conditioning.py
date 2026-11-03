"""Frozen failed-gate replay and train-only conditioning interventions; no updates."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import DecoderData,DecoderModel
from benchmarks.v5.fair_comparison.adapted_models import points
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha

def stats(x):
 x=x.detach().float();return dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),absmean=float(x.abs().mean()),maximum=float(x.abs().max()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True);a=p.parse_args();r=a.root;s=json.loads((a.study/'settings.json').read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=DecoderData(s['cache'],'sam_native');tasks=json.loads((a.study/'learning/sam_native/contract.json').read_text())['tasks'];assert len(tasks)==32
 m=DecoderModel('sam_native');ck=a.study/'learning/sam_native/checkpoint.pt';m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model']);m.eval();before=tree_sha(m.state_dict());rows=[];packed=[];prior_maps=[];start=time.monotonic()
 old=np.load(a.study/'learning/sam_native/masks.npz')['masks']
 def raw_difference(se,qe,prompts):
  se=m.matcher.project(se);qe=F.normalize(m.matcher.project(qe),dim=1);values=[]
  for sign in ['positive','negative']:
   desc=F.normalize(sample_assignment_at_cells(se,prompts[sign+'_xy']),dim=1);sim=torch.einsum('bcp,bchw->bphw',desc,qe);v=prompts[sign+'_mask'].bool();values.append((sim*v[:,:,None,None]).sum(1)/v.sum(1)[:,None,None])
  return values[0]-values[1]
 def decode(qe,prompts,local,prior):
  masks=F.interpolate(prior[:,None],size=(256,256),mode='bilinear',align_corners=False)
  sparse,dense=m.sam.prompt_encoder(points=points(prompts) if local else None,boxes=None,masks=masks)
  out,_=m.sam.mask_decoder(image_embeddings=qe[0],image_pe=m.sam.prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
  logits=m.sam.postprocess_masks(out,(1024,1024),(512,512))[:,0]
  return logits,dense
 for i,task in enumerate(tasks):
  assert data.rows[task['support_id']]['split']==data.rows[task['query_id']]['split']=='train'
  inputs=data.model_inputs(task,'xcell','cuda:0');prompts=inputs['prompts'];swapped=dict(prompts)
  for k in ['xy','mask']:swapped['positive_'+k],swapped['negative_'+k]=swapped['negative_'+k],swapped['positive_'+k]
  with torch.no_grad():
   se=m.encode(inputs['support']);qe=se if inputs['shared'] else m.encode(inputs['query']);zero=qe[0].new_zeros(1,512,512);base=dict(pixel_logits=zero,pixel_probability=zero+.5)
   prior=m.matcher(se[1],qe[1],prompts,base)['pixel_logits'];other=m.matcher(se[1],qe[1],swapped,base)['pixel_logits'];raw=raw_difference(se[1],qe[1],prompts);raw_swap=raw_difference(se[1],qe[1],swapped)
   logits,dense=decode(qe,prompts,inputs['local'],prior);same_points_swap,_=decode(qe,prompts,inputs['local'],other);swlog,swden=decode(qe,swapped,inputs['local'],other)
   base0,den0=decode(qe,prompts,inputs['local'],zero);negative,_=decode(qe,prompts,inputs['local'],-prior);rawprior=F.interpolate((4*raw)[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0];rawlog,_=decode(qe,prompts,inputs['local'],rawprior);rawneg,_=decode(qe,prompts,inputs['local'],-rawprior)
   mask=np.packbits((logits[0]>=0).cpu().numpy());assert np.array_equal(mask,old[i]), 'Frozen replay must equal original saved mask'
   # GT accessed only after all inference and interventions are fixed.
   gt,_=data.labels(task['query_id']);valid=torch.from_numpy(gt!=255).to(logits.device)
   row=dict(plan=task,raw_signed_similarity=stats(raw),raw_swap_change=stats(raw-raw_swap),prior=stats(prior),prior_swap_change=stats(prior-other),dense_swap_change=stats(dense-swden),dense_change_vs_zero=stats(dense-den0),logit_swap_change=stats(logits-swlog),probability_swap_valid_mean=float((logits.sigmoid()-swlog.sigmoid())[0][valid].abs().mean()),prior_only_swap_probability_change=float((logits.sigmoid()-same_points_swap.sigmoid())[0][valid].abs().mean()),zero_prior_probability_change=float((logits.sigmoid()-base0.sigmoid())[0][valid].abs().mean()),negate_prior_probability_change=float((logits.sigmoid()-negative.sigmoid())[0][valid].abs().mean()),raw_signed_prior_probability_swap=float((rawlog.sigmoid()-rawneg.sigmoid())[0][valid].abs().mean()))
   packed.append(mask);prior_maps.append(torch.stack([F.interpolate(prior[:,None],size=(64,64),mode='bilinear',align_corners=False)[0,0],F.interpolate(other[:,None],size=(64,64),mode='bilinear',align_corners=False)[0,0],raw[0]]).cpu().numpy())
  if task['kind']=='cross_present':
   # Differentiate through the actual trained decoder only; no optimizer or parameter update.
   variable=prior.detach().requires_grad_(True);pred,_=decode(qe,prompts,False,variable);truth=torch.from_numpy(gt==task['target']).to(pred.device).float();loss=F.binary_cross_entropy_with_logits(pred[0][valid],truth[valid]);grad=torch.autograd.grad(loss,variable)[0];row['loss_gradient_to_prior']=stats(grad)
  rows.append(row);status=dict(stage='diagnosing',tasks=i+1,total=32,worker_pid=os.getpid(),seconds=time.monotonic()-start);(r/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
 assert tree_sha(m.state_dict())==before
 groups={}
 for domain in ['crc','tiger']:
  for kind in ['local','cross_present','cross_absent']:
   subset=[x for x in rows if x['plan']['dataset']==domain and x['plan']['kind']==kind];g={}
   for field in ['probability_swap_valid_mean','prior_only_swap_probability_change','zero_prior_probability_change','negate_prior_probability_change','raw_signed_prior_probability_swap']:g[field]=float(np.median([x[field] for x in subset]))
   for field in ['raw_signed_similarity','raw_swap_change','prior','prior_swap_change','dense_swap_change','logit_swap_change']:g[field]={key:float(np.median([x[field][key] for x in subset])) for key in subset[0][field]}
   groups[domain+'|'+kind]=g
 np.savez_compressed(r/'diagnostic_arrays.npz',original_masks=np.stack(packed),prior_original_swapped_raw=np.stack(prior_maps))
 report=dict(status='accepted_frozen_train_conditioning_diagnostic',groups=groups,rows=rows,train_only=True,optimizer_updates=0,model_state_unchanged=True,original_32_masks_exact=True,checkpoint_sha256=sha(ck),source_sha256=sha(__file__),seconds=time.monotonic()-start)
 (r/'report.json').write_text(json.dumps(report,indent=2)+'\n');(r/'status.json').write_text(json.dumps(dict(stage='complete',worker_pid=os.getpid())))
if __name__=='__main__':main()
