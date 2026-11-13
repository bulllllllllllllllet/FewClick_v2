"""Preregister only two authorized final-method ablations and immutable inputs."""
import argparse,json,shutil,subprocess,datetime
from pathlib import Path
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;r.mkdir(exist_ok=False)
 original=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/hg_combination_1112_115539/contract.json');c=json.load(open(original))
 freeze=json.load(open('/nfs-medical3/zyh/fewclick_v2/cvpr_research/hg_mix_frozen_1112_163751/method_freeze.json'))
 for f,h in freeze['sources'].items():assert sha(f)==h,('canonical source changed',f)
 files=set(c['sources'])|{str(p) for p in Path('benchmarks/v5/hg_paper_ablation').glob('*.py')}
 c['sources']={f:sha(f) for f in sorted(files)}
 # Only actual public initialization inputs, no historical trained target checkpoint.
 c['weights']={f:h for f,h in c['weights'].items() if 'model_weights/phikon-v2/' in f or 'deeplabv3_resnet50_coco' in f}
 for f,h in c['weights'].items():assert sha(f)==h
 for k in ['training_consumption','functional_source','monitor_source']:assert sha(c[k])==c[k+'_sha256']
 for k in ['mechanism','previous_engineering_failure','extra_functional_gate','matching_recipe','diagnostic_sampling','diagnostic_tasks_sha256']:c.pop(k,None)
 c.update(arms=['M','no_cross_scale'],gpus={'M':3,'no_cross_scale':4},eval_gpus={'M':[3,0],'no_cross_scale':[4,1]},architecture='M: HG-mix minus H; no_cross_scale: HG-mix minus hierarchy and decoder cross-scale region edges, preserve three physical views/7 encoder calls/H',resource_override='User2026-11-13 authorizes prioritized two-arm ablation after WSI complete; at most2train/4eval, all6 eligible, >=12288MiB sharing',status='preregistered',created_at=datetime.datetime.now().astimezone().isoformat(),canonical_method_manifest='/nfs-medical3/zyh/fewclick_v2/cvpr_research/hg_mix_frozen_1112_163751/method_freeze.json',extra_functional_gate='no_cross_scale requires actual H residual>1e-6; M has no H',recipe_diff='Only stated structural interventions. Removed hierarchy parameters preserved frozen for same initialization, not optimized.',repository_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 dump(r/'contract.json',c)
 for arm in c['arms']:(r/arm).mkdir()
 for f in c['sources']:
  dst=r/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);assert sha(dst)==c['sources'][f]
 dump(r/'preparation.json',dict(status='accepted_preregistration',source_count=len(c['sources']),training_tasks=9500,formal_updates=2375,no_old_target_checkpoint_initialization=True))
 print(r,flush=True)
if __name__=='__main__':main()
