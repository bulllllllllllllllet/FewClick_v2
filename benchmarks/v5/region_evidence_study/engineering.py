import argparse,json
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData,RegionMatcher
from benchmarks.v5.support_match_study.model import DecoderModel as Original
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 m=DecoderModel('region_evidence');o=Original('match');assert all(torch.equal(v,o.state_dict()[k]) for k,v in m.state_dict().items());data=DecoderData(Path(c['cache']),'region_evidence');plans=json.load(open(c['functional_source']));results=[]
 with torch.no_grad():
  for d,k in sorted({(t['dataset'],t['kind']) for t in plans}):
   t=next(t for t in plans if (t['dataset'],t['kind'])==(d,k));x=data.model_inputs(t,'xcell','cuda:0');a=m(**x);b=o(**x);assert torch.equal(a['query']['pixel_probability'],b['query']['pixel_probability']);assert torch.isfinite(episode_loss(a,data,t,x['prompts']))
   se=m.encode(m.base.transform(x['support']));qe=se if x['shared'] else m.encode(m.base.transform(x['query']));cached=m.cached_decode(se,qe,x['prompts'],x['local'])[0];assert torch.equal(cached['pixel_probability'],a['query']['pixel_probability']);results.append([d,k])
 # Nonzero output checks ensure regional evidence really affects output and receives feature gradients.
 matcher=RegionMatcher().cuda();torch.nn.init.normal_(matcher.output.weight,std=.1)
 s=torch.randn(1,256,8,8,device='cuda',requires_grad=True);q=torch.randn_like(s);pr={'positive_xy':torch.tensor([[[.2,.3]]],device='cuda'),'negative_xy':torch.tensor([[[.7,.8]]],device='cuda'),'positive_mask':torch.ones(1,1,dtype=torch.bool,device='cuda'),'negative_mask':torch.ones(1,1,dtype=torch.bool,device='cuda')};base={'pixel_logits':torch.zeros(1,32,32,device='cuda'),'pixel_probability':torch.full((1,32,32),.5,device='cuda')};prob=torch.rand(1,32,32,device='cuda',requires_grad=True)
 a=matcher(s,q,pr,base,prob);b=matcher(s,q,pr,base,1-prob);assert (a['pixel_logits']-b['pixel_logits']).abs().max()>0;a['pixel_logits'].sum().backward();assert s.grad.abs().sum()>0 and prob.grad is None
 dump(r/'engineering.json',dict(status='accepted',groups=results,initial_state_and_predictions_exact=True,cached_full_exact=True,region_probability_dependency=True,descriptor_feature_gradient=True,probability_weights_detached=True))
if __name__=='__main__':main()
