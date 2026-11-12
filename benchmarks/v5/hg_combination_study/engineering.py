import argparse,json,os,time
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData
from benchmarks.v5.parallel_phikon_study.model import DecoderModel as Previous,TiledFineBackbone
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.same_slide_retrain.model import frozen_backbone_state
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True);a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));start=time.monotonic()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=DecoderModel(a.arm);parent=Previous('G' if a.arm=='HG' else 'H');d=DecoderData(Path(c['cache']),a.arm);plans=json.load(open(c['functional_source']));checks=[]
 for k,v in parent.state_dict().items():
  if '.backbone.' not in k:assert torch.equal(v,m.state_dict()[k]),k
 frozen=tree_sha(frozen_backbone_state(m))
 with torch.no_grad():
  for domain,kind in sorted({(t['dataset'],t['kind']) for t in plans}):
   t=next(t for t in plans if (t['dataset'],t['kind'])==(domain,kind));x=d.model_inputs(t,'xcell','cuda:0');out=m(**x);assert torch.isfinite(episode_loss(out,d,t,x['prompts']))
   if a.arm=='HG':assert torch.equal(out['query']['pixel_probability'],parent(**x)['query']['pixel_probability'])
   bb=m.base.core.encoders[1].backbone;calls=[];hook=bb.encoder.register_forward_hook(lambda mod,args,result:calls.append(tuple(result.last_hidden_state.shape)))
   try:se=m.encode(m.base.transform(x['support']))
   finally:hook.remove()
   assert len(calls)==(6 if a.arm=='HG' else 7)
   for index in [1,2]:
    image=x['support']['image_'+str(index)];assert torch.equal(m.base.core.encoders[index].backbone(image)['out'],parent.base.core.encoders[index].backbone(image)['out'])
   image=x['support']['image'];actual=m.base.core.encoders[0].backbone(image)['out'];tile=TiledFineBackbone(bb)(image)['out'];expected=tile if a.arm=='HG' else .5*(bb(image)['out']+tile);assert torch.equal(actual,expected)
   qe=se if x['shared'] else m.encode(m.base.transform(x['query']));assert torch.equal(m.cached_decode(se,qe,x['prompts'],x['local'])[0]['pixel_probability'],out['query']['pixel_probability'])
   checks.append(dict(group=[domain,kind],encoder_calls=len(calls),fine_formula_exact=True,context_exact=True));print('passed',domain,kind,flush=True)
 del parent;torch.cuda.empty_cache();t=next(t for t in plans if t['kind']=='same_present');x=d.model_inputs(t,'xcell','cuda:0');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);grads=[]
 for step in range(3):
  opt.zero_grad(set_to_none=True);loss=episode_loss(m(**x),d,t,x['prompts']);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);grads.append({k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None and k in ['pixel.project.weight','extension.head.0.weight','extension.head.2.weight']});opt.step()
 assert all(v>0 for v in grads[-1].values());assert tree_sha(frozen_backbone_state(m))==frozen
 with torch.no_grad():
  se=m.encode(m.base.transform(x['support']));qe=m.encode(m.base.transform(x['query']));out=m.cached_decode(se,qe,x['prompts'],False,with_support=False)[0];assert torch.equal(out['pixel_probability'],m(**x)['query']['pixel_probability'])
  swapped=dict(x['prompts'])
  for k in ['xy','mask']:swapped['positive_'+k],swapped['negative_'+k]=swapped['negative_'+k],swapped['positive_'+k]
  other=m.cached_decode(se,qe,swapped,False,with_support=False)[0];dependency=float((out['pixel_probability']-other['pixel_probability']).abs().mean());assert dependency>0
 dump(r/a.arm/'engineering.json',dict(status='accepted',pid=os.getpid(),checks=checks,HG_initial_G_prediction_exact=a.arm=='HG',cached_full_before_after_exact=True,gradients=grads,support_swap_mean=dependency,frozen_unchanged=True,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),seconds=time.monotonic()-start))
if __name__=='__main__':main()
