"""Count unique model Parameters on CPU using the frozen constructors."""
import json,gc
from pathlib import Path
import torch
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');O=R/'paper_figures_1114_140000';c=json.loads((R/'paper_inference_1114_121131/contract.json').read_text());result=json.loads((O/'parameters.json').read_text()) if (O/'parameters.json').exists() else {}
for method,rec in c['checkpoints'].items():
 if method in result:continue
 if method in ['sam2','wsi']:
  import importlib
  for name in ['benchmarks.v5.additional_sam.model','benchmarks.v5.sam_matched_adaptation.model']:
   module=importlib.import_module(name);source=Path(module.__file__).read_text();exec(compile(source.replace("'cuda:0'","'cpu'"),module.__file__,'exec'),module.__dict__)

 if method=='HG_mix':
  from benchmarks.v5.hg_combination_study.model import DecoderModel
  m=DecoderModel('HG_mix',device='cpu')
 elif method=='sansa':
  # Only relax the device assertion for CPU parameter accounting; architecture/freeze logic unchanged.
  p=Path('benchmarks/v5/sansa_adaptation/model.py');source=p.read_text();assert "and device=='cuda:0'" in source
  scope={'__name__':'cpu_parameter_audit'};exec(compile(source.replace("and device=='cuda:0'","and device=='cpu'"),str(p),'exec'),scope);m=scope['DecoderModel'](device='cpu')
 else:
  from benchmarks.v5.whole_slide_sam.worker import load
  m=load(method,rec['path'],'cpu')
 if method in ['HG_mix','sansa']:m.load_state_dict(torch.load(rec['path'],map_location='cpu',weights_only=True)['model'],strict=True)
 named=list(m.named_parameters());result[method]=dict(total=sum(p.numel() for n,p in named),trainable=sum(p.numel() for n,p in named if p.requires_grad),checkpoint=rec,parameters=[dict(name=n,numel=p.numel(),trainable=p.requires_grad) for n,p in named]);print(method,result[method]['total'],result[method]['trainable'],flush=True)
 (O/'parameters.json').write_text(json.dumps(result,indent=2));del m;gc.collect()
