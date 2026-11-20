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
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');O=Path('论文/CVPR_HG_mix/figures/method_v3_1114');O.mkdir(parents=True,exist_ok=True)
c=json.load(open(R/'paper_inference_1114_121131/contract.json'));data=DecoderData(Path(c['cache']),'match');ev=R/'hg_combination_1112_115539/HG_mix/evaluation';rows={**data.rows,**json.load(open(ev/'query_rows.json'))};t=json.load(open(ev/'tasks.json'))[2675]
def get(id,k):return np.asarray(Image.fromarray(native_context(rows[id],k)).resize((256,256)))
s=[get(t['support_id'],i) for i in range(3)];q=[get(t['query_id'],i) for i in range(3)]
a=np.load(ev/'HG_mix_1/masks.npy',mmap_mode='r');rr=json.load(open(ev/'HG_mix_1/rows.json'));j=next(j for j,r in enumerate(rr) if r['index']==2675);mask=np.unpackbits(a[j],count=512*512).reshape(512,512);mask=np.array(Image.fromarray(mask).resize((256,256),Image.Resampling.NEAREST)).astype(bool);pred=q[0].copy();pred[mask]=(pred[mask]*.5+np.array([37,172,160])*.5).astype('uint8')
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42});fig=plt.figure(figsize=(20,7.6));ax=fig.add_axes([.01,.02,.98,.96]);ax.set_xlim(0,200);ax.set_ylim(0,86);ax.axis('off');ink='#263746';blue='#4b87ae';green='#3e9882';orange='#cb854d'
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
for x in [38,91]:ax.plot([x,x],[7,77],color='#d9e0e4',lw=1)
txt(1,83,'FewClick',22,'left',bold=True);txt(31,83,'One support click set → same-slide remote tissue segmentation',14,'left')
for x,title,col in [(1,'1  Multiscale visual evidence',blue),(41,'2  Hierarchical region context',green),(94,'3  Two complementary support-to-query paths',green)]:txt(x,75,title,11,'left',col,True)
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
# Two explicit branches. Top: region + point logits; bottom: dense probability transport.
# Compact colored lanes, open boundaries rather than text boxes.
ax.add_patch(Rectangle((94,44),105,27,fc='#f0f8f3',ec='none',zorder=0))
ax.add_patch(Rectangle((94,7),105,34,fc='#fcf5ec',ec='none',zorder=0))
txt(96,69,'A  Region prediction + signed point correction',11,'left',green,True)
im(regions(s[0],4),96,49,13);clicks(96,49,13);txt(102.5,46.5,'Support regions',8)
im(regions(q[0],9),120,49,13);txt(126.5,46.5,'Query regions',8)
# Task-conditioned support -> query, distinct from the dense affinity path below.
for sy,qy,col in [(52,59,green),(56,52,orange),(60,60,blue)]:
 ar(108,sy,121,qy,col,1,rad=.08)
txt(115,64,'q⁺, q⁻ → q',10,color=green)
ar(134,55.5,140,55.5,green)
initial=gaussian_filter(mask.astype(float),12)
im(initial,141,50,11);txt(146.5,47,'p₀(Q)*',10)
txt(140,63,'Region + point decoding',8,'left',green)
ar(153,55.5,166,55.5,green);txt(159,59,'logit',9,color=green)
# The summation is explicit and precedes sigmoid.
ax.add_patch(Circle((170,55.5),2.1,fc='white',ec=green,lw=1.6,zorder=9));txt(170,55.5,'+',19,color=green)
ar(172.5,55.5,178,55.5,green);txt(175,59,'σ',12,color=green)
im(pred,180,49,13);txt(186.5,46,'Final p(Q)',9)
txt(177,65,'ℓ₀(Q) + Δℓ(Q)',12,color=green,bold=True)
# Lower branch: support decoder -> anchored probability, coarse prior -> fine affinity.
txt(96,39,'B  Coarse-to-fine dense probability transfer',11,'left',orange,True)
# Support probability illustration independent of the query raster.
yy,xx=np.mgrid[:256,:256];support_prob=np.exp(-((xx-100)**2+(yy-140)**2)/(2*55**2))
im(support_prob,96,20,10);txt(101,17,'pₛ*',10)
ar(102,48,102,41.5,green,style='--');ar(102,36,102,31,orange)
txt(104,34,'Shared support decoder',8,'left',green)
txt(109,23,'+ / −',12,color=orange);ar(106.5,25,114,25,orange)
im(gaussian_filter(support_prob,4),115,20,10);txt(120,17,'Click-anchored p̃ₛ*',8)
# Separate image-feature correspondence matrices, with actual grid sizes labeled.
lattice(132,28,4,1.1,orange);lattice(145,27,5,1.1,orange)
txt(134,25.5,'16×16',8);txt(147.5,24,'32×32',8)
ar(136.5,30,144,30,orange);txt(140.5,33,'coarse prior',7,color=orange)
txt(137,36,'S / Q feature correspondence',8)
ar(125,25,145,27,orange,rad=.1)
im(gaussian_filter(mask.astype(float),8),154,20,10);txt(159,17,'Transferred p*',8)
ar(151,29,154,26,orange)
# Residual head receives both the transferred evidence and current query prediction.
ax.add_patch(Polygon([[168,20],[168,30],[175,28],[175,22]],closed=True,fc='#f3ddc6',ec=orange,lw=1.2,zorder=5))
txt(171.5,25,'h',13,color=orange);ar(164.5,25,167.5,25,orange)
# Query p0 routed around the top lane to the refinement head.
ax.plot([146.5,146.5,166,166],[49,43,43,29],color=green,lw=1.1,zorder=7);ar(166,29,168,28,green)
txt(160,41,'p₀ + point evidence',8,color=green)
ar(175.5,25,182,25,orange);txt(179,29,'2 tanh',8,color=orange)
res=gaussian_filter(mask.astype(float),5)-gaussian_filter(mask.astype(float),20)
im(res,183,20,10);txt(188,17,'Δℓ(Q)*',10,color=orange)
# Unambiguous residual addition: lower output to plus node, NOT to final prediction.
ax.plot([188,196,196,170],[31,35,43,43],color=orange,lw=1.5,zorder=7);ar(170,43,170,53,orange)
txt(149,10,'Dense branch refines the regional result in logit space',10,color=orange,bold=True)
txt(100,1,'Real histology and final prediction; region colors, correspondence links and * intermediate maps are mechanism illustrations.',8)
ar(89,54,95,54,green)
for ext in ['png','svg','pdf']:fig.savefig(O/f'FewClick_method_v3.{ext}',dpi=200,facecolor='white')
(O/'README.md').write_text('''# 方法主图v3（审阅版）
采用真实同心1×/2×/4×输入；整幅/四子块、空间区域及支持→查询对应为主要视觉元素。保留原AAAI的多尺度图像和区域表达。真实样本/最终预测来自原任务2675；彩色区域、对应线及中间概率/残差图仅为机制示意，图下注明，不作为模型可视化证据。上路明确生成p0并转换为logits；下路传输支持预测概率，结合p0及点证据产生2tanh有界残差。残差在sigmoid前与初始logits相加。正文暂未替换。
''');print(O)
