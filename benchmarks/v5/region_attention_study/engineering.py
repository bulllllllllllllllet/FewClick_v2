import argparse,json,os
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData
from benchmarks.v5.region_evidence_study.model import EvidenceModel
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=DecoderModel(a.arm);parent=EvidenceModel();assert all(torch.equal(v,m.state_dict()[k]) for k,v in parent.state_dict().items());d=DecoderData(Path(c['cache']),a.arm);plans=json.load(open(c['functional_source']));groups=[]
 with torch.no_grad():
  for domain,kind in sorted({(t['dataset'],t['kind']) for t in plans}):
   t=next(t for t in plans if (t['dataset'],t['kind'])==(domain,kind));x=d.model_inputs(t,'xcell','cuda:0');out=m(**x);old=parent(**x);assert torch.equal(out['query']['pixel_probability'],old['query']['pixel_probability']);assert torch.isfinite(episode_loss(out,d,t,x['prompts']))
   se=m.encode(m.base.transform(x['support']));qe=se if x['shared'] else m.encode(m.base.transform(x['query']));assert torch.equal(m.cached_decode(se,qe,x['prompts'],x['local'])[0]['pixel_probability'],out['query']['pixel_probability']);groups.append([domain,kind])
 del parent;torch.cuda.empty_cache();t=next(t for t in plans if t['kind']=='same_present');x=d.model_inputs(t,'xcell','cuda:0');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);grad=[]
 for step in range(3):
  opt.zero_grad(set_to_none=True);loss=episode_loss(m(**x),d,t,x['prompts']);loss.backward();z={k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None and (k=='memory.project.weight' or k=='memory.head.2.weight' or k=='refiner.context.weight' or k=='refiner.head.4.weight')};assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);grad.append(z);opt.step()
 assert grad[-1]['memory.project.weight']>0 and grad[-1]['memory.head.2.weight']>0
 if a.arm=='C':assert grad[-1]['refiner.context.weight']>0 and grad[-1]['refiner.head.4.weight']>0
 with torch.no_grad():
  se=m.encode(m.base.transform(x['support']));qe=m.encode(m.base.transform(x['query']));out=m.cached_decode(se,qe,x['prompts'],False)[0];assert torch.equal(out['pixel_probability'],m(**x)['query']['pixel_probability'])
  # Attention after real updates must respond to support content.
  task=m.base.core.support_task(se,x['prompts']);src=m.base.core.decode(se,task,x['prompts'],True)[0];a1,ctx=m.memory(se['pixel_embedding'],qe['pixel_embedding'],x['prompts'],src['pixel_probability']);a2,_=m.memory(-se['pixel_embedding'],qe['pixel_embedding'],x['prompts'],src['pixel_probability']);assert (a1-a2).abs().max()>0
 dump(r/a.arm/'engineering.json',dict(status='accepted',pid=os.getpid(),groups=groups,parent_state_initial_outputs_exact=True,cached_full_before_after_exact=True,gradients=grad,support_dependency=True,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))
if __name__=='__main__':main()
