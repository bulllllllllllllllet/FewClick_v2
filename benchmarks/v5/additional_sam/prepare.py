import argparse,json,hashlib,shutil
from pathlib import Path
import yaml
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump
ROOT=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/additional_sam_1108_135227')
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['sam2_public','wsi_sam_public']);a=p.parse_args();r=ROOT/a.method;r.mkdir(exist_ok=False);src=ROOT.parent/'wsi_same_slide_1108_013913';old=json.load(open(src/'contract.json'));accepted=json.load(open(src/'acceptance.json'));assert accepted['tasks_sha256']==sha(src/'tasks.json')
 for f in ['tasks.json','query_rows.json','coverage.json']:shutil.copy2(src/f,r/f)
 if a.method=='wsi_sam_public':
  spec=yaml.safe_load(Path('benchmarks/v4/baseline/configs/wsi_sam_zero_shot.yaml').read_text());weights=list(spec['checkpoints'].values())
 else:weights=[dict(path=str(ROOT/'sam2.1_hiera_large.pt'),sha256=sha(ROOT/'sam2.1_hiera_large.pt'))]
 for w in weights:assert sha(w['path'])==w['sha256']
 signature=hashlib.sha256(json.dumps(weights,sort_keys=True).encode()).hexdigest();old.update(checkpoints={a.method:dict(weights=weights,sha256=signature)},sources={str(p):sha(p) for p in Path('benchmarks/v5/additional_sam').glob('*.py')},comparison_track='PUBLIC_WEIGHTS_0_TARGET_UPDATES + explicit source-feature signed-cosine dense-query-mask + native decoder; not matched target-supervision to trained SAM bridge',method=a.method)
 dump(r/'contract.json',old)
if __name__=='__main__':main()
