import argparse,json,os,gc
from pathlib import Path
import torch
from .model import DecoderModel,frozen_decoder_state
from benchmarks.v5.sam_matched_adaptation.model import DecoderModel as Old,DecoderData
from benchmarks.v5.architecture_bridge.train_monitor import fixed_train_plans
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'contract.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 old=Old('wsi_sam_ft');old.eval();expected=tree_sha(old.state_dict());data=DecoderData(Path(c['cache']),'wsi_sam_ft');plans=fixed_train_plans(Path(c['training_source']));selected=[next(t for t in plans if t['dataset']==d and t['kind']==k) for d in ['crc','tiger'] for k in ['local','cross_present','cross_absent']]
 with torch.no_grad():refs=[old(**data.model_inputs(t,'xcell','cuda:0'))['query']['pixel_logits'].cpu() for t in selected]
 old_trainable={n for n,p in old.named_parameters() if p.requires_grad};del old;gc.collect();torch.cuda.empty_cache();new=DecoderModel();new.eval();assert tree_sha(new.state_dict())==expected
 with torch.no_grad():
  for t,ref in zip(selected,refs):assert torch.equal(new(**data.model_inputs(t,'xcell','cuda:0'))['query']['pixel_logits'].cpu(),ref)
 trainable={n for n,p in new.named_parameters() if p.requires_grad};assert trainable<old_trainable;removed=old_trainable-trainable;assert all(n.startswith(('wrapped.net_high.','wrapped.net_low.')) for n in removed)
 dump(r/'integration_acceptance.json',dict(status='complete_accepted_initial_state_and_six_predictions_exact',worker_pid=os.getpid(),initial_full_state_sha256=expected,only_decoder_parameter_mask_changed=True,removed_trainable_names=sorted(removed),trainable_numel=sum(p.numel() for p in new.parameters() if p.requires_grad),frozen_decoder_sha256=tree_sha(frozen_decoder_state(new))))
if __name__=='__main__':main()
