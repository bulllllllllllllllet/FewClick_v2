"""Audit full ROI GT registration without selecting inference inputs from labels."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,pyvips,yaml
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;_,catalog=load_cell_catalog(TASKS)
 source=Path('/nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet');cohort=pd.read_parquet(source).set_index('wsi_id').to_dict('index');plan=json.load(open('/nfs-medical3/zyh/fewclick_v2/cvpr_research/paper_completion_dirfix_1106_193345/wsi_grid_plan.json'))
 palette=yaml.safe_load(Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml').read_text())['data']['class_map'];rows=[]
 for i,p in enumerate(plan['plans']):
  sid=p['slide'].split('|')[1];c=cohort[sid];path=Path(c['gt_path']);gt=pyvips.Image.new_from_file(str(path),access='random');wsi=pyvips.Image.new_from_file(p['geometry']['source_image'],access='random');box=p['tiles'][0]['boxes'][0];fw,fh=box[2:];x0,y0=box[:2];x1=x0+2*fw;y1=y0+2*fh
  sx=gt.width/wsi.width;sy=gt.height/wsi.height;gx0=int(np.floor(x0*sx));gy0=int(np.floor(y0*sy));gx1=int(np.ceil(x1*sx));gy1=int(np.ceil(y1*sy));assert 0<=gx0<gx1<=gt.width and 0<=gy0<gy1<=gt.height
  crop=gt.crop(gx0,gy0,gx1-gx0,gy1-gy0).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(crop.height,crop.width,3);lab=np.full(rgb.shape[:2],255,np.uint8)
  for item in palette:lab[(rgb==item['rgb']).all(-1)]=item['id']
  xs=np.floor((x0+(np.arange(1024)+.5)*(x1-x0)/1024)*sx).astype(int)-gx0;ys=np.floor((y0+(np.arange(1024)+.5)*(y1-y0)/1024)*sy).astype(int)-gy0;labels=lab[ys[:,None],xs[None,:]]
  with Image.open(catalog[p['task']['query_id']]['labels']) as im:old=np.asarray(im)
  center=labels[256:768,256:768];agreement=float((old==center).mean());Image.fromarray(labels).save(r/f'roi_{i}_labels.png')
  # Save comparison only; no admission threshold or alignment tuning from GT.
  Image.fromarray(np.concatenate([old,center,(old!=center).astype('uint8')*255],axis=1)).save(r/f'roi_{i}_center_audit.png')
  rows.append(dict(slide=sid,gt_path=str(path),GT_size=[gt.width,gt.height],WSI_size=[wsi.width,wsi.height],GT_to_native_scale=[1/sx,1/sy],ROI_native_xyxy=[x0,y0,x1,y1],center_label_agreement=agreement,valid_fraction=float((labels!=255).mean()),label_file=str(r/f'roi_{i}_labels.png'),label_sha256=sha(r/f'roi_{i}_labels.png')));print(json.dumps(rows[-1]),flush=True)
 (r/'annotation_audit.json').write_text(json.dumps(dict(status='complete_registration_audit_acceptance_pending',source_manifest_sha256=sha(source),rows=rows,limitation='dimension-ratio affine candidate; center replay discrepancies must be examined before full ROI scores are accepted; no query GT inference or model selection'),indent=2))
if __name__=='__main__':main()
