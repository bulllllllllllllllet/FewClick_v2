"""Read-only coverage and signed support diagnostics; no model or threshold selection."""
import json,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from .prepare import dump

def main():
 r=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/wsi_diagnostic_1108_005817');src=r.parent/'wsi_extent_1107_221801';_,catalog=load_cell_catalog(TASKS);counts=Counter();slides=defaultdict(set)
 for row in catalog.values():
  if row['dataset']=='crc' and row['split']=='val':
   for k in json.loads(row['eligible_json']):counts[int(k)]+=1;slides[int(k)].add(row['space_key'])
 dump(r/'coverage.json',dict(eligible_val_fields=dict(counts),source_slides={k:len(v) for k,v in slides.items()},limitation='eligible fields overlap in class membership, not independent new cases; no safe continuous expansion guarantee'))
 plans=json.load(open(src/'plan.json'))['plans'];results=[]
 for item in plans:
  if item['task']['target'] not in [0,7,10]:continue
  i=item['index'];task=item['task'];target=task['target'];gt=np.asarray(Image.open(src/f'labels_{i:03}.png'));support=np.asarray(Image.open(catalog[task['support_id']]['labels']));clicks={}
  for sign in ['positive','negative']:
   clicks[sign]=[int(support[min(511,int(y*512)),min(511,int(x*512))]) for x,y in task[sign+'_xy']]
  assert all(v==target for v in clicks['positive']) and all(v!=target and v!=255 for v in clicks['negative']), (i,clicks)
  row=dict(index=i,target=target,slide=item['slide'],support_target_fraction=float((support==target).mean()),support_click_labels=clicks,GT_target_fraction=float((gt==target).mean()),methods={})
  for method in ['full','no_match','sam']:
   path=src/f'{method}_{i%2}'/f'roi_{i:03}.npz'
   with np.load(path) as z:pred=z['probability']>=.5
   valid=gt!=255;truth=gt==target;tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
   row['methods'][method]=dict(precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else None,dice=2*tp/(2*tp+fp+fn) if tp+fn else None,GT_positive=tp+fn,predicted=tp+fp,FP_by_GT_class={int(k):int(v) for k,v in zip(*np.unique(gt[pred&~truth&valid],return_counts=True))},probability_sha256=sha(path))
  panels=[]
  for name in [f'rgb_{i:03}',f'gt_{i:03}',f'full_{i:03}',f'no_match_{i:03}',f'sam_{i:03}']:
   panels.append(np.asarray(Image.open(src/'gallery_v2'/f'{name}.png').resize((384,384))))
  Image.fromarray(np.concatenate(panels,axis=1)).save(r/f'case_{i:03}.png');results.append(row);print(json.dumps(row),flush=True)
 dump(r/'diagnosis.json',dict(status='complete_readonly_counts_and_support_label_checks',rows=results,source_plan_sha256=sha(src/'plan.json'),training_updates=0,limitation='8fixedROI diagnostic; causal hypotheses require separate intervention, no queryGT threshold/model selection'))
 page='<meta charset="utf-8"><h1>组织低分诊断</h1><p>原图 / GT / 完整FewClick / 去匹配 / SAM桥；支持点击在原gallery另见。只读诊断，不改阈值。</p>'
 for row in results:page+=f'<p>{row["slide"]} target{row["target"]}</p><img width="1500" src="case_{row["index"]:03}.png">'
 (r/'index.html').write_text(page)
if __name__=='__main__':main()
