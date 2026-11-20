"""Editable HG-mix method overview; illustrative tokens, real histology inputs."""
from pathlib import Path
import json,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Circle,FancyArrowPatch
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.support_match_study.model import DecoderData
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');O=Path('论文/CVPR_HG_mix/figures/method_v1_1114');O.mkdir(parents=True,exist_ok=True)
c=json.load(open(R/'paper_inference_1114_121131/contract.json'));d=DecoderData(Path(c['cache']),'match');ev=R/'hg_combination_1112_115539/HG_mix/evaluation';rows={**d.rows,**json.load(open(ev/'query_rows.json'))};t=json.load(open(ev/'tasks.json'))[2675]
s=np.asarray(Image.fromarray(native_context(rows[t['support_id']],0)).resize((512,512)));q=np.asarray(Image.fromarray(native_context(rows[t['query_id']],0)).resize((512,512)))
a=np.load(ev/'HG_mix_1/masks.npy',mmap_mode='r');rr=json.load(open(ev/'HG_mix_1/rows.json'));j=next(j for j,r in enumerate(rr) if r['index']==2675);mask=np.unpackbits(a[j],count=512*512).reshape(512,512).astype(bool);pred=q.copy();pred[mask]=(q[mask]*.5+np.array([45,170,190])*.5).astype('uint8')
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(20,11));ax=fig.add_axes([.015,.02,.97,.96]);ax.set_xlim(0,180);ax.set_ylim(0,98);ax.axis('off')
ink='#243449';blue='#4387b1';green='#439783';orange='#c68144';purple='#8d6caf'
def text(x,y,s,size=10,weight='normal',ha='center',color=ink):ax.text(x,y,s,fontsize=size,weight=weight,ha=ha,va='center',color=color)
def box(x,y,w,h,s='',fc='white',ec='#adb8c3',size=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15,rounding_size=0.8',fc=fc,ec=ec,lw=1));text(x+w/2,y+h/2,s,size)
def arrow(x,y,X,Y,color=ink,style='-',rad=0):ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=11,lw=1.25,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))
def panel(x,y,w,h,title,color,fill):
 box(x,y,w,h,fc=fill,ec=color);text(x+2,y+h-3,title,12,'bold','left',color)
def im(img,x,y,w,h=None):
 h=w if h is None else h;ax.imshow(img,extent=(x,x+w,y,y+h),zorder=3);ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='#8393a1',lw=.7,zorder=4))
def tok(x,y,n=6,color=green):
 for k in range(n):ax.add_patch(Rectangle((x+k*1.7,y),1.3,2,fc=color,ec='white',lw=.4))
def grid(x,y,n=5,color=blue):
 for i in range(n):
  for j in range(n):ax.add_patch(Rectangle((x+i*1.25,y+j*1.25),1.15,1.15,fc=color,alpha=.35+.55*((i+j)%3)/2,ec='white',lw=.3))
text(1,95,'FewClick',22,'bold','left');text(30,95,'One support click set → remote tissue segmentation across the same WSI',14,ha='left')
# top overview
panel(0,55,34,35,'(a) Same-slide support & query',blue,'#f2f7fb')
im(s,2,65,12);im(q,19,65,12)
for sign,color,mark in [('positive','#19e264','+'),('negative','#19d8ed','x')]:
 xy=np.array(t[sign+'_xy']);ax.scatter(2+xy[:,0]*12,77-xy[:,1]*12,c=color,marker=mark,s=40,linewidths=1.7,zorder=8)
text(8,61.5,'Support + clicks',9);text(25,61.5,'Remote query',9);text(17,57.5,'Aligned 1× / 2× / 4× fields of view',9)
panel(38,55,39,35,'(b) Shared visual encoding',blue,'#f2f7fb')
box(42,73,31,9,'Frozen Phikon-v2',fc='#dcebf5',ec=blue,size=12)
box(42,60,31,8,'Whole / tiled fine fusion\n+ two contextual views',fc='white',ec=blue)
arrow(57.5,73,57.5,68,blue)
panel(81,55,44,35,'(c) Prompt-conditioned regions',green,'#f1f8f4')
tok(85,77,6);tok(85,73,6);tok(85,69,6);text(111,74,'Hierarchical\nregion tokens',10)
box(85,58,36,8,'Signed prompt encoding\n+ region & point decoding',fc='white',ec=green)
arrow(103,69,103,66,green)
panel(129,55,51,35,'(d) Dense transfer & output',orange,'#fdf6ee')
box(132,71,25,11,'Coarse-to-fine\nprobability transfer',fc='white',ec=orange)
im(pred,162,65,15);text(169.5,61,'Query prediction',9)
box(132,59,25,8,'Bounded residual\nrefinement',fc='white',ec=orange);arrow(144,71,144,67,orange);arrow(157,63,160,70,orange)
for x,X in [(34,38),(77,81),(125,129)]:arrow(x,74,X,74)
# lower expanded modules
panel(0,4,55,45,'(b) Whole–tiled fusion',blue,'#f7fafc')
im(q,3,25,12);text(9,22,'Fine field',9)
# show whole and four tiles
im(q,20,33,7)
for iy in range(2):
 for ix in range(2):im(q[iy*256:(iy+1)*256,ix*256:(ix+1)*256],20+ix*4,24-iy*4,3.6)
arrow(15,32,19,36,blue);arrow(15,29,19,26,blue)
text(24,17,'4 crops → resize → encode',8)
box(31,30,12,9,'Shared\nencoder',fc='#dcebf5',ec=blue,size=9)
arrow(27,36.5,31,36,blue);arrow(28,24,31,32,blue)
grid(47,31,4);arrow(43,35,46,35,blue)
text(46,27,'0.5 + 0.5',10,'bold');text(27.5,12,'Align whole features + stitch four tile features',9)
text(27.5,7,'Fine 64×64 grid  •  2× and 4× context encoded separately',8)
panel(59,4,57,45,'(c) Hierarchical regions & signed prompts',green,'#f5faf7')
# tree tokens
for y,n in [(34,3),(27,5),(20,7)]:tok(63,y,n)
for y,n in [(34,3),(27,5)]:
 for k in range(n):arrow(64+k*1.7,y,64+k*2,y-4,green)
text(75,38,'4×',8);text(75,31,'2×',8);text(77,24,'1×',8)
text(70,16,'Soft assignment\n& sparse scale links',8)
box(84,32,28,8,'Positive / negative\nset encoder',ec=green,fc='#e1f1e7',size=9)
text(86,28,'q⁺',11,color=green);text(94,28,'q⁻',11,color='#c86165');text(106,28,'→ q',11)
arrow(77,34,83,36,green)
box(84,17,28,8,'Region decoding\n+ signed point similarity',ec=green,size=9);arrow(98,27,98,25,green);arrow(77,21,84,21,green)
text(88,9,'Support probability pₛ  /  query logits ℓ₀',9)
panel(120,4,60,45,'(d) Coarse-to-fine probability transfer',orange,'#fdf8f3')
box(123,33,21,8,'Support probability\n+ click anchors',ec=orange,size=9)
box(148,33,28,8,'Support / query\nprojected features',ec=orange,size=9)
grid(126,21,4,orange);text(132,18,'16×16 matching',8);grid(149,21,5,orange);text(156,18,'32×32 matching',8)
arrow(162,33,153,28,orange);arrow(157,33,131,28,orange);arrow(132,24,148,24,orange)
text(141,27,'prior',8,color=orange)
arrow(133,33,133,29,orange);arrow(156,20,156,14,orange)
box(123,7,53,7,'Transport pₛ  →  Δℓ = 2 tanh(head)  →  σ(ℓ₀ + Δℓ)',ec=orange,size=9)
# explanatory footnote
text(90,1,'Same support reused across query windows; overlapping predictions are fused in WSI coordinates.  Tokens / affinities are schematic.',9)
for ext in ['svg','pdf','png']:fig.savefig(O/f'FewClick_method_v1.{ext}',dpi=200,facecolor='white')
(O/'README.md').write_text('''# 方法主图 v1（待审阅）

原AAAI：保留真实病理输入、token层级和按模块分区的视觉表达；按当前HG-mix替换内容。
参考布局：FiVE (CVPR 2024) Fig.2 的总览/局部展开；FOCUS (CVPR 2025) Fig.1 的子模块分区和token对应示意。
https://openaccess.thecvf.com/content/CVPR2024/html/Li_Generalizable_Whole_Slide_Image_Classification_with_Fine-Grained_Visual-Semantic_Interaction_CVPR_2024_paper.html
https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_FOCUS_Knowledge-enhanced_Adaptive_Visual_Compression_for_Few-shot_Whole_Slide_Image_CVPR_2025_paper.pdf

输入/预测：原11000任务2675，HG_mix_0已保存mask。图中token和对应示意是结构示意，不是注意力可视化。SVG保留可编辑文字，PDF用于排版，PNG用于预览。本稿待用户确认，未替换正文图1。
''')
print(O)
