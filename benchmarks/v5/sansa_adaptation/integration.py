import argparse,json,os
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import DecoderModel,DecoderData,frozen_backbone_state
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=DecoderModel();m.eval();trainable=[n for n,p in m.named_parameters() if p.requires_grad];assert len(trainable)==240 and all('adapter' in n for n in trainable);data=DecoderData(Path(c['cache']),'sansa_ft');plans=fixed_train_plans(Path(c['training_source']));selected=[next(t for t in plans if t['dataset']==d and t['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']];before=tree_sha(m.state_dict())
 with torch.no_grad():
  for t in selected:
   x=data.model_inputs(t,'xcell','cuda:0');frames=[m.encode(x['support'])]
   if not x['local']:frames.append(m.encode(x['query']))
   pp=x['prompts'];xy=torch.cat([pp[s+'_xy'][0][pp[s+'_mask'][0]] for s in ['positive','negative']])[None]*640;labels=torch.tensor([[1]*int(pp['positive_mask'].sum())+[0]*int(pp['negative_mask'].sum())],device='cuda:0',dtype=torch.int32);prompt={0:{0:dict(prompt_type='point',prompt=dict(point_coords=xy,point_labels=labels))},'shots':1};native=m.network(torch.stack(frames)[None],prompt)['pred_masks'];native=F.interpolate(native[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0];out=m(**x)
   assert torch.equal(native[-1][None],out['query']['pixel_logits']);assert torch.equal(native[0][None],out['support']['pixel_logits'])
 assert tree_sha(m.state_dict())==before
 dump(r/'integration_acceptance.json',dict(status='complete_accepted_six_native_train_outputs_exact',worker_pid=os.getpid(),initial_state_sha256=before,trainable_names=trainable,trainable_numel=sum(p.numel() for p in m.parameters() if p.requires_grad),frozen_nonadapter_encoder_sha256=tree_sha(frozen_backbone_state(m)),official_presence_disabled=not m.network.sam.pred_obj_scores))
if __name__=='__main__':main()
