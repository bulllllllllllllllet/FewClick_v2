"""Real-input matched initialization and checkpoint identity gate."""
import argparse,json
from pathlib import Path
import torch
from .model import DecoderModel,DecoderData
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.load(open(r/'study.json'));torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=DecoderData(Path(c['cache']),'control');a=DecoderModel('control');b=DecoderModel('region');a.eval();b.eval();assert tree_sha(a.state_dict())==tree_sha(b.state_dict());assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
 tasks=[json.loads(x) for x in Path(c['training_source']).read_text().splitlines()];selected=[]
 for d in ['crc','tiger']:
  for k in ['local','cross_present','cross_absent']:selected.append(next(t for t in tasks if t['dataset']==d and t['kind']==k))
 with torch.no_grad():
  for plan in selected:
   x=data.model_inputs(plan,'xcell','cuda:0');left=a(**x);right=b(**x);assert torch.equal(left['query']['pixel_logits'],right['query']['pixel_logits']);assert torch.equal(left['support']['pixel_logits'],right['support']['pixel_logits'])
 (r/'preflight.json').write_text(json.dumps(dict(status='accepted',same_parameters=True,initial_logits_equal_tasks=6,initial_state_sha256=tree_sha(a.state_dict())))+'\n')
if __name__=='__main__':main()
