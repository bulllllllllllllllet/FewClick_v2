import argparse,json,os,gc
from pathlib import Path
import torch
from .model import DecoderModel,auto_points
from benchmarks.v5.wsi_author_freeze.model import DecoderModel as Old
from benchmarks.v5.sam_matched_adaptation.model import DecoderData
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 # Actual GPU automatic prompt selection/rejection/extrema/NMS checks.
 for value in [0.,-10.]:assert auto_points(torch.full((1,1,256,256),value,device='cuda:0'))[0] is None
 synthetic=torch.full((1,1,256,256),-10.,device='cuda:0');synthetic[...,60:100,60:100]=10.;a,trace=auto_points(synthetic);assert a is not None and sum(trace['labels'])>0;assert auto_points(synthetic)[1]==trace
 old=Old();old.eval();state=tree_sha(old.state_dict());data=DecoderData(Path(c['cache']),'wsi_sam_ft');plans=fixed_train_plans(Path(c['training_source']));selected=[next(t for t in plans if t['dataset']==d and t['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']]
 with torch.no_grad():refs=[old(**data.model_inputs(t,'xcell','cuda:0'))['query']['pixel_logits'].cpu() for t in selected]
 del old;gc.collect();torch.cuda.empty_cache();new=DecoderModel();new.eval();assert tree_sha(new.state_dict())==state;traces=[]
 with torch.no_grad():
  for t,ref in zip(selected,refs):
   x=data.model_inputs(t,'xcell','cuda:0');new.auto_enabled=False;assert torch.equal(new(**x)['query']['pixel_logits'].cpu(),ref);new.auto_enabled=True;out=new(**x)['query']['pixel_logits'];assert torch.isfinite(out).all()
   if x['local']:assert torch.equal(out.cpu(),ref)
   else:
    trace=new.last_auto_points;assert all(0<u<1 and 0<v<1 for u,v in trace['xy']);assert sum(trace['labels'])<=3 and len(trace['labels'])-sum(trace['labels'])<=2;traces.append(dict(dataset=t['dataset'],kind=t['kind'],**trace))
 assert tree_sha(new.state_dict())==state
 dump(r/'integration_acceptance.json',dict(status='complete_accepted_dense_control_local_exact_auto_points_valid',worker_pid=os.getpid(),initial_state_sha256=state,dense_disabled_remote_exact=True,local_exact=True,query_GT_prompt_selection=False,synthetic_rejection_and_NMS_replay=True,traces=traces))
if __name__=='__main__':main()
