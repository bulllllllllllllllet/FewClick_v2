import argparse,json
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData
from .loss import matching_loss,episode_loss,WEIGHT
from benchmarks.v5.support_match_study.model import DecoderModel as Parent
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss as old_loss
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();c=json.load(open(a.root/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=DecoderModel(a.arm);parent=Parent('match');assert all(torch.equal(v,parent.state_dict()[k]) for k,v in m.state_dict().items());d=DecoderData(Path(c['cache']),a.arm);plans=json.load(open(c['functional_source']));groups=[]
 for domain,kind in sorted({(t['dataset'],t['kind']) for t in plans}):
  t=next(t for t in plans if (t['dataset'],t['kind'])==(domain,kind));x=d.model_inputs(t,'xcell','cuda:0')
  out=m(**x);aux=matching_loss(out,d,t);assert torch.isfinite(aux) and aux>0
  named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];grads=torch.autograd.grad(aux,[p for _,p in named],retain_graph=True,allow_unused=True)
  values={n:float(g.norm()) for (n,p),g in zip(named,grads) if g is not None};assert values['pixel.project.weight']>0;assert all(n.startswith('pixel.project.') or value==0 for n,value in values.items())
  total=episode_loss(out,d,t,x['prompts']);assert torch.equal(total,old_loss(out,d,t,x['prompts'])+WEIGHT*aux)
  with torch.no_grad():
   ref=parent(**x);pred=m(**x);assert torch.equal(out['query']['pixel_probability'],ref['query']['pixel_probability']) and torch.equal(pred['query']['pixel_probability'],ref['query']['pixel_probability'])
  groups.append(dict(dataset=domain,kind=kind,aux=float(aux.detach()),gradients=values))
 dump(a.root/a.arm/'engineering.json',dict(status='accepted',initial_state_predictions_exact=True,training_only=True,extra_parameters=0,aux_grad_only_matching_projection=True,groups=groups))
if __name__=='__main__':main()
