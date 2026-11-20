"""Visual method schematic: real multiscale images, illustrative regions/correspondences."""
from pathlib import Path
import json,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Circle,Polygon
from scipy.ndimage import gaussian_filter
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.support_match_study.model import DecoderData
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');O=Path('论文/CVPR_HG_mix/figures/method_v2_1114');O.mkdir(parents=True,exist_ok=True)
c=json.load(open(R/'paper_inference_1114_121131/contract.json'));data=DecoderData(Path(c['cache']),'match');ev=R/'hg_combination_1112_115539/HG_mix/evaluation';rows={**data.rows,**json.load(open(ev/'query_rows.json'))};t=json.load(open(ev/'tasks.json'))[2675]
def get(id,k):return np.asarray(Image.fromarray(native_context(rows[id],k)).resize((256,256)))
s=[get(t['support_id'],i) for i in range(3)];q=[get(t['query_id'],i) for i in range(3)]
a=np.load(ev/'HG_mix_1/masks.npy',mmap_mode='r');rr=json.load(open(ev/'HG_mix_1/rows.json'));j=next(j for j,r in enumerate(rr) if r['index']==2675);mask=np.unpackbits(a[j],count=512*512).reshape(512,512);mask=np.array(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST)).astype(bool);pred=q[0].copy();pred[mask]=(pred[mask]*.5+np.array([37,172,160])*.5).astype('uint8')
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42});fig=plt.figure(figsize=(20,9));ax=fig.add_axes([.01,.02,.98,.96]);ax.set_xlim(0,200);ax.set_ylim(0,86);ax.axis('off');ink='#263746';blue='#4b87ae';green='#3e9882';orange='#cb854d'
def txt(x,y,z,size=10,ha='center',color=ink,bold=False):ax.text(x,y,z,fontsize=size,ha=ha,va='center',color=color,weight='bold' if bold else 'normal',zorder=10)
def ar(x,y,X,Y,col=ink,lw=1.3,rad=0,style='-'):ax.add_patch(FancyArrowPatch((x,y),(X,Y),arrowstyle='-|>',mutation_scale=11,lw=lw,color=col,linestyle=style,connectionstyle=f'arc3,rad={rad}',zorder=7))
def im(a,x,y,w,h=None):
 h=w if h is None else h;ax.imshow(a,extent=(x,x+w,y,y+h),zorder=3);ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#a6adb4',lw=.6,zorder=4))
def frame(x,y,w,h,col):ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=col,lw=1.6,zorder=6))
def clicks(x,y,w):
 for sign,col,m in [('positive','#22df62','+'),('negative','#15d9ec','x')]:
  v=np.array(t[sign+'_xy']);ax.scatter(x+v[:,0]*w,y+(1-v[:,1])*w,c=col,marker=m,s=45,linewidths=1.7,zorder=12)
def lattice(x,y,n,step=1.35,col=blue):
 for i in range(n):
  for j in range(n):ax.add_patch(Rectangle((x+i*step,y+j*step),step*.88,step*.88,fc=col,alpha=.25+.65*((i*3+j*2)%7)/6,ec='white',lw=.3,zorder=4))
def regions(image,seed):
 rng=np.random.default_rng(seed);yy,xx=np.mgrid[:256,:256];centers=np.array([(i*64+32,j*64+32) for i in range(4) for j in range(4)])+rng.normal(0,12,(16,2));dist=np.array([(yy-y)**2+(xx-x)**2 for y,x in centers]);lab=dist.argmin(0);colors=np.array(plt.get_cmap('Pastel1').colors);rgb=colors[lab%len(colors)]*255;out=.38*image+.62*rgb;edges=np.zeros(lab.shape,bool);edges[1:]|=lab[1:]!=lab[:-1];edges[:,1:]|=lab[:,1:]!=lab[:,:-1];out[edges]=[100,115,120];return out.astype('uint8')
# page headings and light column separators
for x in [38,91,145]:ax.plot([x,x],[7,77],color='#d9e0e4',lw=1)
txt(1,83,'FewClick',22,'left',bold=True);txt(31,83,'One support click set → same-slide remote tissue segmentation',14,'left')
for x,title,col in [(1,'1  Multiscale visual evidence',blue),(41,'2  Hierarchical region context',green),(94,'3  Support-to-query correspondence',orange),(148,'4  Pixel refinement',green)]:txt(x,75,title,11,'left',col,True)
# left actual nested context for two branches
for imgs,y,name in [(s,46,'Support'),(q,12,'Query')]:
 im(imgs[2],2,y,18);frame(6.5,y+4.5,9,9,blue);frame(8.75,y+6.75,4.5,4.5,orange)
 im(imgs[1],23,y+10,11);frame(25.75,y+12.75,5.5,5.5,orange)
 im(imgs[0],23,y-3,11)
 ar(20,y+13,22,y+16,blue);ar(20,y+7,22,y+2,orange)
 txt(11,y-3,'4× context',9);txt(28.5,y+23,'2×',9);txt(28.5,y-6,'1× fine',9);txt(2,y+22,name,12,'left',bold=True)
 if name=='Support':clicks(23,y-3,11)
# central upper fine fusion visual
im(s[0],42,57,11);txt(47.5,54,'Whole',9)
for iy in range(2):
 for ix in range(2):im(s[0][iy*128:(iy+1)*128,ix*128:(ix+1)*128],42+ix*5.8,38+(1-iy)*5.8,5.2)
txt(47.5,35,'Four resized tiles',9)
# shared encoder as trapezoid
ax.add_patch(Polygon([[57,43],[57,66],[64,62],[64,47]],closed=True,fc='#d9eaf3',ec=blue,lw=1));txt(60.5,54,'E',17,color=blue,bold=True);txt(61,40,'shared / frozen',8)
ar(53,62,56,60,blue);ar(54,44,56,49,blue)
lattice(68,57,4);lattice(68,43,6,1,blue);ar(64,59,67,59,blue);ar(64,49,67,46,blue)
txt(78,53,'⊕',23,color=blue);ar(73.5,59,78,55,blue);ar(74,46,78,51,blue);txt(78,47,'½ + ½',10,color=blue)
im(regions(s[0],4),81,50,8);ar(80,53,81,53,blue);txt(75,69,'Whole–tiled fine fusion',10,color=blue,bold=True)
# region maps and hierarchical graph
for k,(x,y,w) in enumerate([(42,12,13),(50,19,11),(57,25,9)]):im(regions(s[k],k+5),x,y,w)
txt(49,9,'Soft regions',9)
levels=[[(72,15),(77,15),(82,15),(87,15)],[(75,24),(82,24)],[(78.5,32)]]
for low,up in zip(levels,levels[1:]):
 for i,p in enumerate(low):
  target=up[min(len(up)-1,i//2)];ar(*p,*target,green,.9);ar(target[0]+.6,target[1],p[0]+.6,p[1],green,.7,rad=.15)
for level in levels:
 for x,y in level:ax.add_patch(Circle((x,y),1.6,fc='#b9ddc8',ec=green,lw=1,zorder=8))
ar(65,27,71,25,green);txt(78,9,'Cross-scale messages',9);ar(85,49,85,35,green,style='--');txt(42,5,'Phikon-v2  •  64 regions per scale  •  shared S/Q encoding',8,'left')
# correspondence central: support query maps with explicit matched links
im(regions(s[0],4),96,49,17);im(regions(q[0],9),124,49,17);clicks(96,49,17)
txt(104.5,69,'Support regions',10,bold=True);txt(132.5,69,'Query regions',10,bold=True)
for (sy,qy,col) in [(53,60,green),(58,54,orange),(63,63,blue)]:
 ax.add_patch(Circle((110,sy),.65,fc=col,zorder=9));ax.add_patch(Circle((127,qy),.65,fc=col,zorder=9));ar(110,sy,127,qy,col,1.3,rad=.15)
txt(119,44,'Signed prompts → task-conditioned region prediction',8)
# support probability schematic explicitly marked
heat=gaussian_filter(mask.astype(float),8);im(heat,96,18,11);txt(101.5,15,'pₛ + click anchors*',9)
lattice(112,29,4,1.4,orange);lattice(126,26,7,1.4,orange)
ar(118,32,125,32,orange);txt(122,36,'coarse prior',8,color=orange);txt(114.5,26,'16²',8);txt(131,22,'32²',8)
ar(101.5,30,112,32,orange,rad=-.2);ar(113,49,114,36,orange);ar(132,49,132,37,orange)
ar(107,22,125,27,orange,rad=.12);txt(119,10,'Coarse-to-fine probability transport',10,color=orange,bold=True)
# output visual maps plus residual
im(gaussian_filter(mask.astype(float),12),149,48,16);txt(157,69,'Initial query probability*',9,bold=True);im(pred,177,48,16);txt(185,69,'Final prediction',10,bold=True)
ar(141,57,148,57,green);ar(166,56,176,56,green);txt(171,62,'σ',14,color=green)
im(heat,149,21,13);txt(155.5,18,'Transferred evidence*',8)
# residual signed field schematic
res=gaussian_filter(mask.astype(float),5)-gaussian_filter(mask.astype(float),20);im(res,178,21,13);txt(184.5,18,'Bounded residual*',8)
ar(140,28,148,28,orange);ar(163,28,177,28,orange);ar(184,35,184,47,orange);txt(171,33,'2 tanh',10,color=orange);txt(171,10,'Region logits + dense residual',10,color=green,bold=True)
txt(100,1,'Actual histology and final prediction; colored regions, matching links and * intermediate maps illustrate the mechanism, not measured activations.',8)
for ext in ['png','svg','pdf']:fig.savefig(O/f'FewClick_method_v2.{ext}',dpi=200,facecolor='white')
(O/'README.md').write_text('''# 方法主图v2（审阅版）
采用真实同心1×/2×/4×输入；整幅/四子块、空间区域及支持→查询对应为主要视觉元素。保留原AAAI的多尺度图像和区域表达。真实样本/最终预测来自原任务2675；彩色区域、对应线及中间概率/残差图仅为机制示意，图下注明，不作为模型可视化证据。正文暂未替换。
''');print(O)
