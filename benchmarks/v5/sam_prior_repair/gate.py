"""Native SAM remote adapter engineering gate; labels only in training loss."""
import argparse,json,os,time,gc
from pathlib import Path
import torch
from .model import DecoderData,DecoderModel
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();s=json.loads((a.root/'settings.json').read_text());out=a.root/'engineering';out.mkdir()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=DecoderData(s['cache'],'sam_native');plans=fixed_train_plans('/nfs-medical3/zyh/fewclick_v2/cvpr_research/phikon_joint_tasks_1031_222610/data/training4000.jsonl')
 selected=[next(x for x in plans if x['dataset']==d and x['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']]
 m=DecoderModel('sam_native');m.eval();common=tree_sha(m.base.state_dict());backbone=tree_sha(m.sam.image_encoder.state_dict())
 def progress(**v):
  v['worker_pid']=os.getpid();(out/'status.json').write_text(json.dumps(v));print(json.dumps(v),flush=True)
 with torch.no_grad():
  for i,plan in enumerate(selected):
   inputs=data.model_inputs(plan,'xcell','cuda:0');calls=[]
   hook=m.sam.prompt_encoder.register_forward_pre_hook(lambda mod,args,kwargs:calls.append(kwargs['points'] is None),with_kwargs=True)
   result=m(**inputs);hook.remove();assert torch.isfinite(result['query']['pixel_logits']).all()
   assert calls[0]==(not inputs['local']), 'Remote decoder must receive no query point coordinates'
   if not inputs['local']:assert not result['query_prompt_maps'].any()
   assert torch.isfinite(episode_loss(result,data,plan,inputs['prompts']))
   changed={**inputs,'query':{k:(v+17 if k.startswith('cells') or k.startswith('total') else v) for k,v in inputs['query'].items()}}
   assert torch.equal(m(**changed)['query']['pixel_logits'],result['query']['pixel_logits'])
   progress(stage='real_forward',tasks=i+1,total=6)
 m.train();opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=1e-4,weight_decay=1e-4)
 def update(model,optimizer,batch):
  optimizer.zero_grad(set_to_none=True)
  for plan in batch:
   inputs=data.model_inputs(plan,'xcell','cuda:0');loss=episode_loss(model(**inputs),data,plan,inputs['prompts']);(loss/len(batch)).backward()
  assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None);optimizer.step()
 update(m,opt,plans[:4]);update(m,opt,plans[4:8])
 assert m.matcher.project.weight.grad is not None and m.matcher.project.weight.grad.abs().sum()>0
 assert any(x.grad is not None and x.grad.abs().sum()>0 for x in m.sam.mask_decoder.parameters())
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),out/'continuation.pt')
 update(m,opt,plans[8:12]);expected=tree_sha(m.state_dict());expected_opt=tree_sha(opt.state_dict())
 fresh=DecoderModel('sam_native');saved=torch.load(out/'continuation.pt',weights_only=True);fresh.load_state_dict(saved['model']);fresh.train();fopt=torch.optim.AdamW([x for x in fresh.parameters() if x.requires_grad],lr=1e-4,weight_decay=1e-4);fopt.load_state_dict(saved['optimizer']);update(fresh,fopt,plans[8:12]);assert tree_sha(fresh.state_dict())==expected and tree_sha(fopt.state_dict())==expected_opt
 del fresh,fopt,saved;gc.collect();torch.cuda.empty_cache();times=[]
 for i in range(20):
  torch.cuda.synchronize();start=time.monotonic();update(m,opt,[plans[(i*4+j)%32] for j in range(4)]);torch.cuda.synchronize();times.append(time.monotonic()-start);progress(stage='throughput',updates=i+1,total=20)
 assert tree_sha(m.sam.image_encoder.state_dict())==backbone
 infer=[];m.eval()
 with torch.no_grad():
  for plan in (selected*17)[:100]:
   inputs=data.model_inputs(plan,'xcell','cuda:0');torch.cuda.synchronize();start=time.monotonic();m(**inputs);torch.cuda.synchronize();infer.append(time.monotonic()-start)
 b=dict(status='accepted_support_match_GPU_probe',arm='sam_native',updates=20,inference_tasks=100,train_seconds=times,inference_seconds=infer,initial_common_sha256=common,source_sha256=sha(__file__),frozen_backbone=True,query_points_remote_zero=True,exact_optimizer_continuation=True)
 (out/'sam_native_benchmark.json').write_text(json.dumps(b,indent=2)+'\n');(out/'acceptance.json').write_text(json.dumps(dict(status='accepted_sam_matched_engineering',common_initial_sha256=common,benchmark=b),indent=2)+'\n');progress(stage='complete')
if __name__=='__main__':main()
