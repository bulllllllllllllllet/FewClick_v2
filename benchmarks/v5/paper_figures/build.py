"""CPU-only rendering of frozen predictions; no model inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pyvips
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.whole_slide_once.data import rgb_box,grid,boxes_at,gt_extent
from benchmarks.v5.whole_slide_once.worker import palette,intersects
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');O=R/'paper_figures_1114_140000'
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2))
C=read(R/'paper_inference_1114_121131/contract.json');methods=list(C['checkpoints']);names=['FewClick','SANSA','SAM','SAM2','WSI-SAM','SAM-Med2D']
evs={m:Path(v['path']).parent.parent/'evaluation' for m,v in C['checkpoints'].items()}
tasks=read(evs['HG_mix']/'tasks.json');rows=read(evs['HG_mix']/'query_rows.json')
# Support rows are part of the normalized catalog used for frozen tasks.
from benchmarks.v5.support_match_study.model import DecoderData
data=DecoderData(Path(C['cache']),'match');rows={**data.rows,**rows}
metrics={};masks={}
for m in methods:
 metrics[m]={};masks[m]={}
 for shard in range(2):
  p=evs[m]/f'{m}_{shard}';a=np.load(p/'masks.npy',mmap_mode='r')
  for j,r in enumerate(read(p/'rows.json')):metrics[m][r['index']]=r['metrics'];masks[m][r['index']]=(a,j)
# Freeze deterministic selection before image inspection: closest to median HG Dice per band.
bands=[('Large',lambda r:r['positive']/r['valid']>=.2),('Small',lambda r:0<r['positive']/r['valid']<.01),('Absent',lambda r:r['positive']==0),('Failure',lambda r:r['positive']>0 and r['dice']<.2)]
selected=[]
for name,pred in bands:
 ids=[i for i,r in metrics['HG_mix'].items() if pred(r)];key='false_positive_fraction' if name=='Absent' else 'dice';med=float(np.median([metrics['HG_mix'][i][key] for i in ids]));i=min(ids,key=lambda i:(abs(metrics['HG_mix'][i][key]-med),i));selected.append(dict(band=name,index=i,median=med,task=tasks[i],scores={m:metrics[m][i] for m in methods}))
dump(O/'case_selection.json',dict(rule='Within each stated band, closest to HG median Dice (absent: FPA); ties ascending task index. Fixed before rendering; retrospective descriptive selection.',cases=selected))
def points(ax,t):
 for s,c,marker in [('positive','lime','+'),('negative','cyan','x')]:
  a=np.array(t[s+'_xy'])*512;ax.scatter(a[:,0],a[:,1],c=c,marker=marker,s=45,linewidths=1.4)
def overlay(im,mask):
 a=im.astype(float).copy();a[mask]=a[mask]*.55+np.array([240,50,70])*.45;return a.astype('uint8')
fig,axes=plt.subplots(4,9,figsize=(22,10))
for n,s in enumerate(selected):
 i=s['index'];t=tasks[i];support=np.asarray(Image.fromarray(native_context(rows[t['support_id']],0)).resize((512,512)));q=np.asarray(Image.fromarray(native_context(rows[t['query_id']],0)).resize((512,512)));gt=np.array(Image.open(rows[t['query_id']]['labels']));truth=gt==t['target']
 ims=[support,q,overlay(q,truth)]
 for m in methods:
  a,j=masks[m][i];mask=np.unpackbits(a[j],count=512*512).reshape(512,512).astype(bool);ims.append(overlay(q,mask))
 for col,im in enumerate(ims):
  ax=axes[n,col];ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
  title=['Support (+/-)','Query','Ground truth']+names
  if col>=3:
   met=metrics[methods[col-3]][i];title[col]+=f"\n{'FPA' if n==2 else 'Dice'} {100*met['false_positive_fraction' if n==2 else 'dice']:.1f}%"
  ax.set_title(title[col],fontsize=9)
 axes[n,0].set_ylabel(f"{s['band']} | class {t['target']}\ntask {i}",fontsize=10);points(axes[n,0],t)
fig.tight_layout();fig.savefig(O/'figure3.png',dpi=180);fig.savefig(O/'figure3.pdf');plt.close(fig)
print('figure3 complete',flush=True)
# Figure2: first fixed formal slide, its first supported target (no score-based selection).
case=read(R/'whole_wsi_current_1112_171010/cases.json')[0];s=case['supports'][0];target=s['target'];W,H=case['gt_width'],case['gt_height'];step=max(1,int(np.ceil(max(W,H)/1100)));ys=np.arange(0,H,step);xs=np.arange(0,W,step)
gt=pyvips.Image.new_from_file(case['gt_path'],access='random');small=gt.resize(1/step,kernel='nearest').extract_band(0,n=3);rgb=np.frombuffer(small.write_to_memory(),np.uint8).reshape(small.height,small.width,3);labels=np.full(rgb.shape[:2],255,np.uint8)
# Decode exact sampled GT pixels below, aligned to packed prediction indices.
# Use exact GT index samples via narrow rows to align with saved mask sampling.
ys=np.minimum((np.arange(small.height)*step),H-1);xs=np.minimum(np.arange(small.width)*step,W-1)
rgb=np.stack([np.frombuffer(gt.crop(0,int(y),W,1).extract_band(0,n=3).write_to_memory(),np.uint8).reshape(W,3)[xs] for y in ys])
for color in palette():labels[(rgb==color['rgb']).all(-1)]=color['id']
thumb=pyvips.Image.thumbnail(case['wsi_path'],small.width,height=small.height,size='force').extract_band(0,n=3);im=np.frombuffer(thumb.write_to_memory(),np.uint8).reshape(small.height,small.width,3)
remote=np.ones(labels.shape,bool);groups,bw,bh=grid(case,case['supports']);sup=s['geometry']['native_boxes'][2]
for y,xx in groups:
 for x in xx:
  if intersects(boxes_at(x,y,bw,bh)[2],sup):
   x0,y0,x1,y1=gt_extent([x,y,bw,bh],case)
   remote[np.ix_((ys>=y0)&(ys<y1),(xs>=x0)&(xs<x1))]=False
report=read(R/'WSI_final15_closeout_1113_224245/report.json');paths=report['source_report_SHA256'];fig,axes=plt.subplots(2,5,figsize=(16,10));ax=axes.flat
support=np.asarray(Image.fromarray(rgb_box(case['wsi_path'],s['geometry']['native_boxes'][0])).resize((512,512)));ax[0].imshow(support);points(ax[0],s);ax[0].set_title('Support (+/-)')
ax[1].imshow(im);x,y,w,h=s['geometry']['native_boxes'][0];ax[1].add_patch(Rectangle((x/case['width']*small.width,y/case['height']*small.height),w/case['width']*small.width,h/case['height']*small.height,fill=False,edgecolor='cyan',linewidth=2));ax[1].set_title('Whole slide / support box')
ax[2].imshow(overlay(im,labels==target));ax[2].set_title('Ground truth')
domain=np.full((*labels.shape,3),245,np.uint8);domain[labels!=255]=[180,180,180];domain[(labels!=255)&remote]=[70,145,190];ax[3].imshow(domain);ax[3].set_title('Valid domain: gray + blue\nStrict remote: blue')
for k,m in enumerate(methods):
 p=next(Path(p) for p in paths if '/'+m+'/' in p and '/case_00/' in p);a=np.load(p.parent/f'target_{target}/mask.npy',mmap_mode='r');mask=np.unpackbits(a[ys],axis=1,count=W)[:,xs].astype(bool);ax[k+4].imshow(overlay(im,mask));rr=read(p);entry=next(v for v in rr['targets'] if v['target']==target) if isinstance(rr['targets'],list) else rr['targets'][str(target)];ax[k+4].set_title(names[k]+f" | Dice {entry['whole_WSI']['dice']*100:.1f}%")
for a in ax:a.set_xticks([]);a.set_yticks([])
fig.tight_layout();fig.savefig(O/'figure2.png',dpi=180);fig.savefig(O/'figure2.pdf');dump(O/'wsi_selection.json',dict(slide=case['wsi_id'],target=target,rule='first formal case, first support target',source_case=case));print('figure2 complete')
