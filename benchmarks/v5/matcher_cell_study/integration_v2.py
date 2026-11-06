"""Real train-only integration, cell perturbation, gradient and initialization checks."""
import argparse,json,copy
from pathlib import Path
import torch
from .model import DecoderData,DecoderModel
from benchmarks.v5.support_match_study.model import DecoderModel as Reference
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=DecoderData(Path(c['cache']),'cell_on',c['seed']);plans=list(map(json.loads,Path(c['training_source']).read_text().splitlines()));chosen=[]
 for domain in ['crc','tiger']:
  for kind in ['local','cross_present','cross_absent']:
   chosen.append(next(p for p in plans if p['dataset']==domain and p['kind']==kind))
 on=DecoderModel('cell_on',c['seed']);off=DecoderModel('cell_off',c['seed']);ref=Reference('match',c['seed']);assert tree_sha(on.state_dict())==tree_sha(off.state_dict())==tree_sha(ref.state_dict())
 checks=[]
 with torch.no_grad():
  for plan in chosen:
   x=data.model_inputs(plan,'xcell','cuda:0');assert all(data.rows[plan[k]]['split']=='train' for k in ['query_id','support_id']);on.eval();off.eval();ref.eval()
   a=off(**x)['query']['pixel_logits'];b=ref(**x)['query']['pixel_logits'];assert torch.equal(a,b)
   encoded=on.base.transform(x['support']);assert encoded['cells'].shape[1]>0 and torch.equal(encoded['cells'][...,:10],x['support']['cells'][...,:10])
   changed={k:v.clone() for k,v in x['support'].items()}
   for suffix in ['','_1','_2']:changed['cells'+suffix][...,10:]=0
   perturb=dict(x,support=changed,query=changed if x['shared'] else x['query'])
   on_delta=float((on(**x)['query']['pixel_logits']-on(**perturb)['query']['pixel_logits']).abs().mean())
   assert torch.equal(off(**x)['query']['pixel_logits'],off(**perturb)['query']['pixel_logits'])
   checks.append(dict(domain=plan['dataset'],kind=plan['kind'],cell_embedding_logit_effect=on_delta))
 before={k:v.clone() for k,v in on.named_parameters() if '.cell.' in k};on.train();opt=torch.optim.AdamW([p for p in on.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
 gradient_seen=False
 for plan in chosen[:4]:
  opt.zero_grad(set_to_none=True);x=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(on(**x),data,plan,x['prompts']);assert torch.isfinite(loss);loss.backward();gradient_seen=gradient_seen or any(p.grad is not None and p.grad.abs().sum()>0 for k,p in on.named_parameters() if '.cell.' in k);opt.step()
 assert gradient_seen
 assert any(not torch.equal(before[k],p) for k,p in on.named_parameters() if k in before)
 on.eval()
 with torch.no_grad():
  plan=chosen[0];x=data.model_inputs(plan,'xcell','cuda:0');changed={k:v.clone() for k,v in x['support'].items()}
  for suffix in ['','_1','_2']:changed['cells'+suffix][...,10:]=0
  trained_effect=float((on(**x)['query']['pixel_logits']-on(**dict(x,support=changed,query=changed if x['shared'] else x['query']))['query']['pixel_logits']).abs().mean());assert trained_effect>0

 (r/'cell_integration.json').write_text(json.dumps(dict(status='accepted_actual_cell_integration',initial_states_identical=True,cell_off_exact_original=True,checks=checks,cell_parameters_updated=True,trained_embedding_logit_effect=trained_effect,quality_accepted=False),indent=2));print('accepted_actual_cell_integration',flush=True)
if __name__=='__main__':main()
