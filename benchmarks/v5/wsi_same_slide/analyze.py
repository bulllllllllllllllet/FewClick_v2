"""Read-only clustered statistics, error gallery, and whole-slide efficiency inventory."""
import argparse,json,hashlib,html,os
from pathlib import Path
from collections import defaultdict
import numpy as np,pyvips
from PIL import Image,ImageDraw
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.paper_multiscale.data import native_context,intersects
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.wsi_quality_study.prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;src=r.parent/'wsi_same_slide_1108_013913';a=json.load(open(src/'acceptance.json'));assert a['report_sha256']==sha(src/'report.json') and a['tasks_sha256']==sha(src/'tasks.json');tasks=json.load(open(src/'tasks.json'));qr=json.load(open(src/'query_rows.json'));_,catalog=load_cell_catalog(TASKS);allrows=dict(catalog,**qr);methods=['full','no_match','sam'];classes=[0,1,3,4,5,6,7,8,9,10,11];slides=sorted({t['query_space'] for t in tasks});si={s:i for i,s in enumerate(slides)};ci={k:i for i,k in enumerate(classes)};metrics={};positions={};disk={}
 for method in methods:
  rows={};pos={}
  for shard in [0,1]:
   out=src/f'{method}_{shard}';acc=json.load(open(out/'acceptance.json'));assert acc['rows_sha256']==sha(out/'rows.json');assert acc['masks_sha256']==sha(out/'masks.npy');assert acc['input_sha256']==sha(out/'input_hashes.json');rr=json.load(open(out/'rows.json'));disk[method,shard]=np.load(out/'masks.npy',mmap_mode='r')
   for j,x in enumerate(rr):rows[x['index']]=x['metrics'];pos[x['index']]=(shard,j)
  assert sorted(rows)==list(range(11000));metrics[method]=rows;positions[method]=pos
 count=np.zeros((len(slides),11));sums={m:np.zeros_like(count) for m in methods};strata=defaultdict(list);per_slide=defaultdict(list)
 for i,t in enumerate(tasks):
  row=qr[t['query_id']];s=catalog[t['support_id']];a=json.loads(s['multiscale_json'])['context_um'];b=json.loads(row['multiscale_json'])['context_um'];distance=float(np.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2));db='<2mm' if distance<2000 else '2-4mm' if distance<4000 else '>=4mm';x=metrics['full'][i];ab='absent' if not x['positive'] else '<1%' if x['positive']/x['valid']<.01 else '1-5%' if x['positive']/x['valid']<.05 else '5-20%' if x['positive']/x['valid']<.2 else '>=20%'
  strata[t['target'],ab,db].append(i);per_slide[t['query_space']].append(i)
  if x['positive']:
   count[si[t['query_space']],ci[t['target']]]+=1
   for m in methods:sums[m][si[t['query_space']],ci[t['target']]]+=metrics[m][i]['dice']
 point={m:float(np.mean(sums[m].sum(0)/count.sum(0))) for m in methods};old=json.load(open(src/'report.json'))
 for m in methods:assert abs(point[m]-old['results'][m]['macro_Dice'])<1e-14
 rng=np.random.default_rng(20261108);w=rng.multinomial(len(slides),np.ones(len(slides))/len(slides),size=2000);den=w@count;assert (den>0).all();boot={m:np.mean((w@sums[m])/den,axis=1) for m in methods};stats={m:dict(macro_Dice=point[m],slide_cluster_CI95=np.quantile(boot[m],[.025,.975]).tolist()) for m in methods}
 differences={m:dict(full_minus_reference=point['full']-point[m],CI95=np.quantile(boot['full']-boot[m],[.025,.975]).tolist()) for m in ['no_match','sam']}
 tables=[]
 for (target,ab,db),indices in sorted(strata.items()):
  res={}
  for m in methods:
   mm=[metrics[m][i] for i in indices];pos=[x for x in mm if x['positive']];res[m]=dict(dice=float(np.mean([x['dice'] for x in pos])) if pos else None,FPA=float(np.mean([x['false_positive_fraction'] for x in mm])) if ab=='absent' else None,empty_prediction_fraction=float(np.mean([x['tp']+x['fp']==0 for x in mm])),precision=float(np.mean([x['tp']/(x['tp']+x['fp']) if x['tp']+x['fp'] else 0 for x in mm])),recall=float(np.mean([x['tp']/x['positive'] for x in pos])) if pos else None)
  tables.append(dict(target=target,area=ab,distance=db,n=len(indices),query_regions=len({tasks[i]['query_id'] for i in indices}),slides=len({tasks[i]['query_space'] for i in indices}),methods=res))
 dump(r/'strata.json',tables);dump(r/'per_slide.json',{s:{m:dict(n=len(ii),present=sum(metrics[m][i]['positive']>0 for i in ii),class_macro_Dice=float(np.mean([np.mean([metrics[m][i]['dice'] for i in ii if tasks[i]['target']==k and metrics[m][i]['positive']]) for k in classes if any(tasks[i]['target']==k and metrics[m][i]['positive'] for i in ii)]))) for m in methods} for s,ii in per_slide.items()})
 selected=[];seen=set()
 def choose(indices,n,reason):
  used=0
  for i in indices:
   if tasks[i]['query_id'] in seen:continue
   selected.append(dict(index=i,reason=reason));seen.add(tasks[i]['query_id']);used+=1
   if used==n:break
 present=[i for i in range(11000) if metrics['full'][i]['positive']]
 choose(sorted([i for i in present if metrics['full'][i]['positive']/metrics['full'][i]['valid']<.01],key=lambda i:(metrics['full'][i]['dice'],i)),8,'small_target_worst_Dice')
 for k in [7,11]:choose(sorted([i for i in present if tasks[i]['target']==k],key=lambda i:(metrics['full'][i]['dice']-metrics['no_match'][i]['dice'],i)),6,f'class{k}_largest_loss_to_no_match')
 choose(sorted([i for i in present if metrics['full'][i]['positive']/metrics['full'][i]['valid']>=.2],key=lambda i:(-metrics['full'][i]['dice'],i)),4,'large_target_best_control');dump(r/'selected_cases.json',selected)
 gallery=r/'gallery';gallery.mkdir();page='<meta charset="utf-8"><h1>同片远端诊断</h1><p>事后诊断选择，未用于选模型或阈值。支持（绿正点/紫负点）｜Query｜GT｜FewClick｜去匹配｜SAM桥。点击图片查看原分辨率。</p><pre>'+html.escape(json.dumps(dict(cluster_statistics=stats,paired_differences=differences),ensure_ascii=False,indent=2))+'</pre>'
 for item in selected:
  i=item['index'];t=tasks[i];q=qr[t['query_id']];gt=np.asarray(Image.open(q['labels']));rgb=np.asarray(Image.fromarray(native_context(q,0)).resize((512,512),Image.Resampling.BILINEAR));support=np.asarray(Image.fromarray(native_context(catalog[t['support_id']],0)).resize((512,512),Image.Resampling.BILINEAR));support=Image.fromarray(support);draw=ImageDraw.Draw(support)
  for sign,color in [('positive','lime'),('negative','magenta')]:
   for x,y in t[sign+'_xy']:draw.ellipse((x*512-4,y*512-4,x*512+4,y*512+4),outline=color,width=2)
  panels=[np.asarray(support),rgb];truth=rgb.copy();mask=gt==t['target'];truth[mask]=(.55*truth[mask]+.45*np.array([255,220,0])).astype('uint8');truth[gt==255]=128;panels.append(truth)
  for m in methods:
   shard,j=positions[m][i];pred=np.unpackbits(disk[m,shard][j],count=512*512).reshape(512,512).astype(bool);assert score(pred,mask,gt!=255,np.array(json.loads(q['input_mpp_xy_json'])))==metrics[m][i];overlay=rgb.copy();overlay[pred]=(.55*overlay[pred]+.45*np.array([255,0,0])).astype('uint8');overlay[gt==255]=128;panels.append(overlay)
  path=gallery/f'case_{i:05}.png';Image.fromarray(np.concatenate(panels,axis=1)).save(path);caption=dict(index=i,target=t['target'],reason=item['reason'],slide=t['query_space'],area=metrics['full'][i]['positive']/metrics['full'][i]['valid'],scores={m:metrics[m][i]['dice'] for m in methods});page+='<p>'+html.escape(json.dumps(caption,ensure_ascii=False))+'</p>'+f'<a href="gallery/{path.name}"><img width="100%" src="gallery/{path.name}"></a>';print(json.dumps(dict(stage='gallery',index=i)),flush=True)
 (r/'index.html').write_text(page)
 # Freeze three source slides without consulting image/label outcomes.
 inventory=[]
 for slide in sorted(slides,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:3]:
  task=next(t for t in tasks if t['query_space']==slide);q=qr[task['query_id']];g=json.loads(q['multiscale_json']);support=catalog[task['support_id']];sf=json.loads(support['multiscale_json'])['context_um'];im=pyvips.Image.new_from_file(g['source_image'],access='random');fw,fh=g['native_boxes'][0][2:];mx,my=g['native_mpp'];train=[json.loads(v['multiscale_json'])['context_um'] for v in catalog.values() if v['split']=='train' and v['space_key']==slide];windows=[]
  for y in range(2*fh,im.height-2*fh+1,fh//2):
   for x in range(2*fw,im.width-2*fw+1,fw//2):
    box=[x-2*fw,y-2*fh,4*fw,4*fh];foot=[box[0]*mx,box[1]*my,(box[0]+box[2])*mx,(box[1]+box[3])*my];safe=not intersects(foot,sf) and not any(intersects(foot,b) for b in train);windows.append(dict(center_native=[x,y],quality_eligible=safe))
  inventory.append(dict(slide=slide,source_image=g['source_image'],source_size=[im.width,im.height],support_task=task,window_count=len(windows),quality_eligible=sum(x['quality_eligible'] for x in windows),boundary_rule='exclude windows lacking full4x; not full-image boundary coverage',windows=windows))
 dump(r/'wsi_efficiency_inventory.json',dict(status='prepared_only_not_executed',slides=inventory,selection='source_slide_SHA first3, first frozen task per slide',quality_rule='train/support overlapping windows engineering only; quality must audit subset and GT registration',missing=['actual encoding and persistent cache','complete-domain update decode/fusion/output','source boundary policy for complete-image claims','cold versus OS-cached IO disclosure']))
 dump(r/'report.json',dict(status='complete_readonly_analysis',cluster_statistics=stats,paired_differences=differences,nslides=len(slides),bootstrap_replicates=2000,strata_rows=len(tables),visual_cases=len(selected),source_tasks_sha256=sha(src/'tasks.json'),limitations=['Within-cohort descriptive development analysis','75slide cluster bootstrap does not establish independent patients or eliminate overlapping spatial dependence','distance and area stratification not causal','posthoc case selection disclosed; no thresholds/weights changed'],WSI_efficiency='inventory_only_not_executed'))
 dump(r/'acceptance.json',dict(status='complete_accepted_readonly_analysis',report_sha256=sha(r/'report.json'),strata_sha256=sha(r/'strata.json'),html_sha256=sha(r/'index.html'),visual_mask_scores_replayed=3*len(selected),worker_pid=os.getpid()))
if __name__=='__main__':main()
