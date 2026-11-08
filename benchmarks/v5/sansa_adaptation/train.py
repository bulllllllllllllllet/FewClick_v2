from .model import frozen_backbone_state
"""Fresh public initialization, frozen recipe, exact8000 consumption; last checkpoint only."""
import argparse,json,os,time,gc
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData,ARMS
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans,evaluate_train_monitor
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();r=a.root;c=json.load(open(r/'contract.json'));arm=r/a.arm
 for path,h in c['sources'].items():assert sha(path)==h,path
 gate=json.load(open(arm/'functional/report.json'));assert gate['status']=='accepted_repaired_functional_gate';assert json.load(open(arm/'functional/parameter_acceptance.json'))['trainable_changed']
 if not a.smoke:assert json.load(open(arm/'smoke/acceptance.json'))['status']=='complete_accepted_smoke'
 out=arm/('smoke' if a.smoke else 'formal');out.mkdir();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 lines=Path(c['training_consumption']).read_text().splitlines(keepends=True);assert len(lines)==8000 and sha(c['training_consumption'])==c['training_consumption_sha256'];steps=4 if a.smoke else 2000
 data=DecoderData(Path(c['cache']),a.arm);m=DecoderModel(a.arm);m.train();initial_trainable=tree_sha({n:p.detach() for n,p in m.named_parameters() if p.requires_grad});frozen=tree_sha(frozen_backbone_state(m));from .model import frozen_decoder_state
 frozen_decoder=tree_sha(frozen_decoder_state(m));opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);plans=fixed_train_plans(Path(c['training_source']));start=time.monotonic();monitors={}
 def status(**v):dump(out/'progress.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,**v));print(json.dumps(dict(pid=os.getpid(),seconds=time.monotonic()-start,**v)),flush=True)
 def monitor(step):
  status(stage='train_monitor',step=step);path=out/f'monitor_{step}.json';evaluate_train_monitor(m,data,plans,'cuda:0',step,path);monitors[str(step)]=sha(path)
 if not a.smoke:monitor(0)
 with (out/'training_consumed.jsonl').open('x') as f:
  for step in range(steps):
   opt.zero_grad(set_to_none=True);loss_sum=0
   for line in lines[step*4:step*4+4]:
    t=json.loads(line);assert data.rows[t['support_id']]['split']==data.rows[t['query_id']]['split']=='train';inputs=data.model_inputs(t,'xcell','cuda:0');loss=episode_loss(m(**inputs),data,t,inputs['prompts']);assert torch.isfinite(loss);(loss/4).backward();loss_sum+=loss.detach().item()/4;f.write(line)
   assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step()
   if step%25==0 or step+1==steps:f.flush();status(stage='training',step=step+1,total=steps,loss=loss_sum)
   if not a.smoke and step+1 in [250,500,1000,2000]:monitor(step+1)
 assert tree_sha(frozen_decoder_state(m))==frozen_decoder,'author-frozen decoder changed'
 assert tree_sha(frozen_backbone_state(m))==frozen;assert tree_sha({n:p.detach() for n,p in m.named_parameters() if p.requires_grad})!=initial_trainable
 torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),updates=steps,seed=20261103),out/'checkpoint.pt');assert (out/'training_consumed.jsonl').read_text()==''.join(lines[:steps*4])
 # Fresh constructor replay on six fixed training groups, including absent/local; no val selection.
 from functools import partial
 allplans=list(map(json.loads,Path(c['training_source']).read_text().splitlines()));selected=[next(t for t in allplans if t['dataset']==d and t['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']];m.eval()
 with torch.no_grad():reference=[m(**data.model_inputs(t,'xcell','cuda:0'))['query']['pixel_probability'].cpu() for t in selected]
 del m,opt;gc.collect();torch.cuda.empty_cache();fresh=DecoderModel(a.arm);fresh.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True,map_location='cuda:0')['model']);fresh.eval()
 with torch.no_grad():
  for t,expected in zip(selected,reference):assert torch.equal(fresh(**data.model_inputs(t,'xcell','cuda:0'))['query']['pixel_probability'].cpu(),expected)
 dump(out/'acceptance.json',dict(status='complete_accepted_smoke' if a.smoke else 'complete_accepted_training',updates=steps,consumed=steps*4,training_consumed_sha256=sha(out/'training_consumed.jsonl'),checkpoint_sha256=sha(out/'checkpoint.pt'),frozen_encoder_sha256=frozen,author_frozen_decoder_sha256=frozen_decoder,actual_trainable_parameters_changed=True,fresh6_predictions_exact=True,monitor_sha256=monitors,arm=a.arm,seed=20261103))
if __name__=='__main__':main()
