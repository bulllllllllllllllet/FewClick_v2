"""Train-only decomposition and counterfactual diagnostics of frozen WSI-SAM."""
import json,os,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from torch.nn import functional as F
from .model import DecoderModel,DecoderData,points
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.architecture_bridge.deterministic_sampling import sample_assignment_at_cells
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.wsi_quality_study.prepare import dump

def run_decode(m,se,qe,prompts,local,mode):
 if mode=='fine_features_only':
  se=dict(se,features=F.normalize(se['native'][0][0],dim=1));qe=dict(qe,features=F.normalize(qe['native'][0][0],dim=1))
 z=qe['features'].new_zeros(1,512,512);residual=m.matcher(se['features'],qe['features'],prompts,dict(pixel_logits=z,pixel_probability=z+.5))['pixel_logits'];source=F.normalize(se['features'].float(),dim=1);query=F.normalize(qe['features'].float(),dim=1);evidence=[]
 for sign in ['positive','negative']:
  desc=F.normalize(sample_assignment_at_cells(source,prompts[sign+'_xy']),dim=1);sim=torch.einsum('bcp,bchw->bphw',desc,query);valid=prompts[sign+'_mask'].bool();evidence.append((sim*valid[:,:,None,None]).sum(1)/valid.sum(1)[:,None,None])
 raw=F.interpolate((4*(evidence[0]-evidence[1]))[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0];prior=raw+(0 if mode=='raw_prior_only' else residual);dense_mask=F.interpolate(prior[:,None],size=(256,256),mode='bilinear',align_corners=False);low=torch.zeros_like(dense_mask);low[...,64:192,64:192]=F.interpolate(dense_mask,size=(128,128),mode='bilinear',align_corners=False);outputs=[]
 for i,(embedding,intermediate) in enumerate(qe['native']):
  sparse,dense=m.wrapped.sam.prompt_encoder(points=points(prompts,low=bool(i)) if local else None,boxes=None,masks=None if mode=='no_dense_prior' else dense_mask if i==0 else low);net=m.wrapped.net_high if i==0 else m.wrapped.net_low;outputs.append(net(image_embeddings=embedding,image_pe=[m.wrapped.sam.prompt_encoder.get_dense_pe()],sparse_prompt_embeddings=[sparse],dense_prompt_embeddings=[dense],multimask_output=False,wsi_token_only=False,interm_embeddings=intermediate))
 mask,_,high,feat=outputs[0];low_token=outputs[1][2];h=(high@feat).view(1,-1,256,256);l=(low_token@feat).view(1,-1,256,256);fused=((high+low_token)@feat).view(1,-1,256,256);logits=mask+fused
 if mode=='high_only':logits=mask+h
 if mode=='sam_residual_only':logits=mask
 if mode=='wsi_fused_only':logits=fused
 logits=F.interpolate(logits,size=(512,512),mode='bilinear',align_corners=False)[:,0]
 stats={name:dict(mean=float(x.mean()),std=float(x.std()),min=float(x.min()),max=float(x.max())) for name,x in [('raw_prior',raw),('learned_prior',residual),('SAM_residual',mask),('high_token_logits',h),('low_token_logits',l),('fused_logits',fused),('final_logits',logits)]}
 return logits,stats

def main():
 r=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/wsi_sam_diagnostic_1108_190545');torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 src=r.parent/'sam_matched_adaptation_1108_142826';c=json.load(open(src/'contract.json'));data=DecoderData(Path(c['cache']),'wsi_sam_ft');plans=fixed_train_plans(Path(c['training_source']));selected=[]
 for d in ['crc','tiger']:
  for k in ['local','cross_present','cross_absent']:selected += [t for t in plans if t['dataset']==d and t['kind']==k][:2]
 assert len(selected)==12;dump(r/'contract.json',dict(tasks=selected,checkpoint_sha256=sha(src/'wsi_sam_ft/formal/checkpoint.pt'),modes=['normal','no_dense_prior','raw_prior_only','high_only','sam_residual_only','wsi_fused_only','fine_features_only'],worker_pid=os.getpid(),training_updates=0,train_only=True))
 m=DecoderModel('wsi_sam_ft');initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()};m.load_state_dict(torch.load(src/'wsi_sam_ft/formal/checkpoint.pt',weights_only=True,map_location='cuda:0')['model']);m.eval();before=tree_sha(m.state_dict());groups=defaultdict(list)
 for n,p in m.named_parameters():
  part='matcher' if n.startswith('matcher.') else 'prompt_encoder' if '.prompt_encoder.' in n else 'net_high' if '.net_high.' in n else 'net_low' if '.net_low.' in n else 'other';delta=(p.detach().cpu()-initial[n]).abs();groups[part].append(dict(name=n,changed=bool(delta.max()>0),max_abs=float(delta.max()),trainable=p.requires_grad))
 dump(r/'parameter_changes.json',dict(groups=groups));del initial;rows=[]
 with torch.no_grad():
  for number,t in enumerate(selected):
   x=data.model_inputs(t,'xcell','cuda:0');se=m.encode(x['support']);qe=se if x['shared'] else m.encode(x['query']);gt,mpp=data.labels(t['query_id'])
   for mode in json.load(open(r/'contract.json'))['modes']:
    logits,stats=run_decode(m,se,qe,x['prompts'],x['local'],mode);assert torch.isfinite(logits).all()
    if mode=='normal':assert torch.equal(logits,m.decode(se,qe,x['prompts'],x['local'])['pixel_logits'])
    pred=(logits[0]>=0).cpu().numpy();metrics=score(pred,gt==t['target'],gt!=255,mpp);rows.append(dict(task=number,dataset=t['dataset'],kind=t['kind'],mode=mode,metrics=metrics,components=stats));print(json.dumps(dict(task=number,mode=mode,dice=metrics['dice'],FPA=metrics['false_positive_fraction'])),flush=True)
 assert tree_sha(m.state_dict())==before;dump(r/'report.json',dict(status='complete_readonly_accepted',rows=rows,weights_unchanged=True,normal_path_exact=True,train_only=True,limitation='Inference counterfactuals on12train tasks, not causal architecture comparison, not validation model selection'))
 dump(r/'acceptance.json',dict(status='complete_accepted_readonly_diagnostic',rows=len(rows),report_sha256=sha(r/'report.json'),weight_state_sha256=before))
if __name__=='__main__':main()
