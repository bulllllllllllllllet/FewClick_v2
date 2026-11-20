"""Native vector reconstruction of approved AAAI-layout concept; no inference."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Polygon,Circle,FancyArrowPatch
import json,hashlib,xml.etree.ElementTree as ET
O=Path('论文/CVPR_HG_mix/figures/method_v5_vector_1114');O.mkdir(parents=True,exist_ok=True)
source=Path('论文/CVPR_HG_mix/figures/method_v4_aaai_1114/FewClick_method_v4_preview.png');photo=Image.open(source);print(photo.size)
# Reference image has normalized 1944 x 810 coordinates. Only histology raster inserts retained.
def crop(box):
 w,h=photo.size;return np.asarray(photo.crop(tuple(round(v*(w/1944 if i%2==0 else h/810)) for i,v in enumerate(box))))
# Use real tissue inserts, preserving the approved composition without baked-in labels.
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.support_match_study.model import DecoderData
import pyvips
R=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');contract=json.load(open(R/'paper_inference_1114_121131/contract.json'));data=DecoderData(Path(contract['cache']),'match');ev=R/'hg_combination_1112_115539/HG_mix/evaluation';rows={**data.rows,**json.load(open(ev/'query_rows.json'))};task=json.load(open(ev/'tasks.json'))[2675]
views=[np.asarray(Image.fromarray(native_context(rows[task['support_id']],i)).resize((256,256))) for i in range(3)]
geom=json.loads(rows[task['support_id']]['multiscale_json']);v=pyvips.Image.thumbnail(geom['source_image'],400).extract_band(0,n=3);wsi=np.frombuffer(v.write_to_memory(),np.uint8).reshape(v.height,v.width,3)
query=np.asarray(Image.fromarray(native_context(rows[task['query_id']],0)).resize((512,512)));rr=json.load(open(ev/'HG_mix_1/rows.json'));idx=next(i for i,r in enumerate(rr) if r['index']==2675);mask=np.unpackbits(np.load(ev/'HG_mix_1/masks.npy',mmap_mode='r')[idx],count=512*512).reshape(512,512).astype(bool);overlay=query.copy();overlay[mask]=(overlay[mask]*.5+np.array([40,165,180])*.5).astype('uint8');imgs=[*views,wsi,views[0],overlay]
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'image.composite_image':False})
fig=plt.figure(figsize=(24,10.2));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1944);ax.set_ylim(826,0);ax.set_aspect('equal');ax.axis('off')
BLUE='#14a7de';GREEN='#28a738';ORANGE='#f28a16';INK='#17252e';uid=0
# Stable descriptive SVG ids keep parts separately selectable.
def ident(obj,name):
 global uid;uid+=1;obj.set_gid(f'{name}_{uid}');return obj
def rect(x,y,w,h,fc='white',ec=INK,lw=1,r=0,name='box',ls='-'):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw,linestyle=ls) if r else Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,linestyle=ls)
 p.set_zorder(4 if fc=='none' else 1);ax.add_patch(ident(p,name));return p
def txt(x,y,s,size=11,ha='center',color=INK,bold=False,name='label'):
 return ident(ax.text(x,y,s,fontsize=size,ha=ha,va='center',color=color,weight='bold' if bold else 'normal',linespacing=1.18,zorder=20),name)
def line(pts,color=INK,lw=1.1,ls='-',name='wire',arrow=True):
 # Straight orthogonal polylines; arrow only at the final segment.
 if len(pts)>2:ident(ax.plot(*zip(*pts[:-1]),color=color,lw=lw,ls=ls,zorder=8)[0],name)
 if arrow:ax.add_patch(ident(FancyArrowPatch(pts[-2],pts[-1],arrowstyle='-|>',mutation_scale=10,lw=lw,color=color,linestyle=ls,zorder=9),name+'_arrow'))
 else:ident(ax.plot(*zip(*pts[-2:]),color=color,lw=lw,ls=ls,zorder=8)[0],name)
def im(a,x,y,w,h,name='histology'):
 ident(ax.imshow(a,extent=(x,x+w,y+h,y),zorder=3),name);rect(x,y,w,h,fc='none',ec='#7c8790',lw=.65)
def panel(x,y,w,h,title,color,fill):
 rect(x,y,w,h,ec=color,lw=1.5,r=10,name='panel');rect(x+1,y+1,w-2,40,fc=fill,ec='none');line([(x,y+42),(x+w,y+42)],color,.8,arrow=False);txt(x+w/2,y+22,title,15,color=color,bold=True)
def box(x,y,w,h,label,size=10,fc='white',ec=INK):
 rect(x,y,w,h,fc=fc,ec=ec,r=4);txt(x+w/2,y+h/2,label,size)
def plus(x,y,r=13,label='+'):
 ax.add_patch(ident(Circle((x,y),r,fc='white',ec=INK,lw=1.3,zorder=12),'addition'));txt(x,y,label,18)
def enc(x,y,w=85,h=65,label='Phikon-v2\nshared / frozen'):
 ax.add_patch(ident(Polygon([(x,y),(x+w,y+12),(x+w,y+h-12),(x,y+h)],fc='#f6fafb',ec=INK,lw=1),'encoder'));txt(x+w*.47,y+h*.48,label,9);txt(x+w*.46,y+h*.82,'E',12,bold=True)
def cube(x,y,w=55,h=44,color='#78acd4'):
 d=13;rect(x,y,w,h,fc=color,ec=color,lw=.6)
 ax.add_patch(ident(Polygon([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],fc='white',ec=color,lw=.7),'feature_top'))
 ax.add_patch(ident(Polygon([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],fc=color,ec=color,lw=.7,alpha=.5),'feature_side'))
 for i in range(1,5):
  line([(x+i*w/5,y),(x+i*w/5,y+h)],'white',.5,arrow=False);line([(x,y+i*h/5),(x+w,y+i*h/5)],'white',.5,arrow=False)
  line([(x+i*w/5,y),(x+i*w/5+d,y-d)],color,.5,arrow=False);line([(x+i*d/5,y-i*d/5),(x+w+i*d/5,y-i*d/5)],color,.5,arrow=False)
def tokens(x,y,w=105,color='#86c4ba',labels=True):
 for row in range(3):
  for col in range(10):rect(x+col*w/10,y+row*23,w/10,12,fc=color,ec='#628b8b',lw=.35,name='token')
 if labels:
  for k,s in enumerate(['R1','R2','R64']):txt(x+w+17,y+6+k*23,s,8)
 txt(x+8,y+39,'⋮',11)
def regions(x,y,w,h,seed=1):
 # Voronoi half-plane clipping creates editable irregular pastel polygons.
 rng=np.random.default_rng(seed);centers=np.array([(i/4+.125,j/4+.125) for j in range(4) for i in range(4)])+rng.uniform(-.07,.07,(16,2));colors=['#c0cae7','#eac4dd','#f2d0ac','#c0d5b0','#d3cae8','#b4d4d0']
 for idx,a in enumerate(centers):
  poly=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
  for b in centers:
   if np.array_equal(a,b):continue
   n=b-a;k=(b@b-a@a)/2;out=[]
   for u,v in zip(poly,np.roll(poly,-1,axis=0)):
    fu=u@n-k;fv=v@n-k
    if fu<=0:out.append(u)
    if (fu<=0)!=(fv<=0):out.append(u+(v-u)*fu/(fu-fv))
   poly=np.array(out)
   if len(poly)==0:break
  if len(poly):ax.add_patch(ident(Polygon(poly*[w,h]+[x,y],fc=colors[idx%6],ec='#7b8589',lw=.4),'soft_region'))
 rect(x,y,w,h,fc='none',ec='#71818a',lw=.7)
def heat(x,y,w,h,seed=1,kind='prob'):
 n=12;yy,xx=np.mgrid[:n,:n];v=np.exp(-((xx-(4+seed%4))**2+(yy-6)**2)/10)+.5*np.exp(-((xx-8)**2+(yy-3)**2)/5)
 if kind=='aff':v=np.exp(-((xx-yy-seed%2)**2)/3)+.15*np.sin(xx*2+yy)
 if kind=='res':v=np.sin(xx*.6)*np.cos(yy*.7)
 v=(v-v.min())/(v.max()-v.min());cm=plt.get_cmap('RdBu_r' if kind=='res' else 'Blues' if kind=='aff' else 'turbo')
 for iy in range(n):
  for ix in range(n):rect(x+ix*w/n,y+iy*h/n,w/n+.05,h/n+.05,fc=cm(v[iy,ix]),ec='none',name='heatmap_cell')
 rect(x,y,w,h,fc='none',ec='#788791',lw=.6)
def click(x,y,sign='+'):
 col='#119645' if sign=='+' else '#e44e53';ax.add_patch(ident(Circle((x,y),10,fc='white',ec=col,lw=1.1,zorder=15),'click'));txt(x,y,sign,15,color=col,bold=True)
panel(6,7,285,808,'1  Multiscale Inputs',BLUE,'#c7e7f5')
panel(301,7,884,535,'2  Visual Encoding & Hierarchical Regions',GREEN,'#d9edc6')
panel(301,552,884,263,'Coarse-to-fine Probability Transfer',ORANGE,'#fce8c8')
panel(1195,7,743,808,'3  Support-conditioned Prediction',ORANGE,'#fbd5bf')
# LEFT: nested image inputs and same-slide support/query.
for k,(yy,name) in enumerate([(81,'Fine\n1×'),(233,'Context\n2×'),(385,'Context\n4×')]):
 txt(58,yy+54,name,12);im(imgs[k],116,yy,145,116)
 if k:rect(170,yy+43,37,30,fc='none',ec='#e54c58',lw=1.2)
 if k<2:
  line([(116,yy+116),(170,yy+152+43)],'#777',.7,'--',arrow=False);line([(261,yy+116),(207,yy+152+43)],'#777',.7,'--',arrow=False)
im(imgs[3],23,548,250,170);rect(84,620,45,40,fc='none',ec='#20a757',lw=1.4);rect(201,637,40,38,fc='none',ec='#5152b5',lw=1.4);txt(105,608,'S',13,bold=True);txt(221,624,'Q',13,bold=True);click(95,634);click(116,649,'−');txt(148,738,'Same-slide support S / query Q',10);txt(148,766,'Positive + / negative − clicks on S',10);txt(148,791,'Aligned physical fields of view',9)
# Fine fusion inset. Distinct whole and tile inputs and distinct encoder passes.
rect(321,63,845,147,fc='#fafcfc',ec='#46565d',r=8,ls='--');txt(337,78,'Fine view: whole + tiled encoding',11,ha='left',bold=True)
im(imgs[0],342,100,75,75);txt(379,190,'Whole',9)
for iy in range(2):
 for ix in range(2):im(imgs[0][iy*128:(iy+1)*128,ix*128:(ix+1)*128],458+ix*38,103+iy*38,34,34)
txt(493,190,'Four tiles',9)
enc(568,97,105,42,'E: whole');enc(568,153,105,42,'E: each tile')
line([(418,120),(435,120),(435,91),(550,91),(550,118),(568,118)],name='whole_encoder_input');line([(532,159),(550,159),(568,174)],name='tiled_encoder_input');txt(911,86,'Phikon-v2: shared / frozen',8)
cube(719,111,48,28,'#86bce3');cube(719,164,48,28,'#f0a0a3');line([(674,118),(718,123)]);line([(674,174),(718,176)]);txt(800,114,'Align',8);txt(800,180,'Stitch',8)
line([(780,124),(863,124),(863,138)]);line([(780,178),(863,178),(863,164)]);txt(827,114,'0.5',9);txt(827,188,'0.5',9);plus(863,151)
line([(877,151),(931,151)]);cube(944,132,63,45,'#c6a4e3');txt(1048,147,'Fused fine\nfeature grid',10);txt(1053,184,'64 × 64',9)
# THREE parallel region rows
for k,y in enumerate([249,341,433]):
 col=['#ef9296','#84add9','#eeb16c'][k];txt(349,y+20,['Fine\n1×','Context\n2×','Context\n4×'][k],10)
 if k==0:
  box(389,y-2,106,44,'Fused features\nfrom inset',9,fc='#f3eafb');line([(976,178),(976,223),(440,223),(440,247)],color='#8e64a6',name='fine_fusion_to_region_row')
 else:enc(389,y-10,106,65,'Phikon-v2\nshared / frozen')
 cube(532,y+4,55,39,col);line([(496,y+20),(531,y+20)]);plus(626,y+20,13,'A');line([(601,y+20),(611,y+20)]);regions(667,y-9,79,64,k+3);line([(640,y+20),(666,y+20)])
 rect(781,y-14,166,77,fc='#e8f6f4',ec='#89b6bb',r=6);tokens(792,y-2,106,['#b9dbcd','#96c6cf','#79a9c6'][k]);line([(747,y+20),(780,y+20)])
 # Explicit input routing from scale images.
 line([(262,[138,290,442][k]),(283,[138,290,442][k]),(283,y+20),(320,y+20)],color=col,ls='--',name='scale_input')
txt(707,232,'Soft regions',9);txt(864,232,'Region tokens (64 / scale)',9)
for y in [306,398]:
 line([(963,y+34),(963,y)],GREEN,name='bottom_up_context');line([(981,y),(981,y+34)],'#437cb6',name='top_down_context')
for yy in [269,361,453]:line([(947,yy),(999,yy)],color='#779b9a',name='regions_to_hierarchical_context')
rect(1001,246,162,245,fc='#f5faf8',ec='#567376',r=7,ls='--');txt(1082,265,'Cross-scale context',10)
for xx,yy in [(1082,292),(1048,326),(1116,326),(1025,360),(1069,360),(1094,360),(1138,360)]:
 rect(xx-12,yy-4,24,8,fc='#8dc0c9',ec='#607f8c',lw=.5)
for a,b in [((1082,296),(1048,322)),((1082,296),(1116,322)),((1048,330),(1025,356)),((1048,330),(1069,356)),((1116,330),(1094,356)),((1116,330),(1138,356))]:line([a,b],color='#7a949c',lw=.7,arrow=False)
tokens(1019,396,96,'#9bbcc9');txt(744,521,'Shared support / query encoding • soft assignment A • hierarchical region features',9)
# QUERY decoder route from center bank, dedicated corridor avoids prompt stage.
line([(1164,416),(1189,416),(1189,307),(1220,307)],name='query_region_tokens')
# RIGHT: prompt encoder chain, q explicitly routes to decoder.
txt(1299,77,'Support S + clicks',10);im(imgs[4],1220,95,130,106);click(1316,120);click(1327,174,'−');regions(1380,95,120,106,9);txt(1440,77,'Prompt → region tokens',10);line([(1351,148),(1378,148)])
box(1530,108,86,80,'Signed set\nencoder',10,fc='#f4f8f4');line([(1501,147),(1528,147)])
for yy,col in [(111,'#a8d9b8'),(158,'#efacb5')]:
 for k in range(7):rect(1641+k*10,yy,10,12,fc=col,ec='#728e82',lw=.4,name='prompt_token')
txt(1728,117,'q⁺',12);txt(1728,164,'q⁻',12)
line([(1617,131),(1638,117)]);line([(1617,164),(1638,164)])
box(1760,118,59,55,'MLP',10);line([(1740,117),(1750,117),(1750,131),(1758,131)]);line([(1740,164),(1750,164),(1750,158),(1758,158)])
rect(1841,135,66,16,fc='#c8d5d7',ec='#829399',lw=.5);txt(1874,114,'Task q',10);line([(1820,144),(1840,144)]);line([(1874,152),(1874,224),(1354,224),(1354,274)],color='#2c7554',name='task_token_to_query_decoder')
# Region prediction -> projection -> point correction -> p0
rect(1220,279,89,72,fc='#edf6f4',ec='#779695',r=3);tokens(1226,286,52,'#a7c8d0');txt(1263,263,'Query R',10)
box(1323,276,108,81,'Context-aware\nregion decoder',10);line([(1310,313),(1322,313)])
box(1452,280,92,73,'Soft region\n→ pixel\nprojection',9);line([(1432,313),(1451,313)])
box(1569,280,93,73,'Signed point\nsimilarity\ncorrection',9);line([(1545,313),(1568,313)])
heat(1690,281,76,72,3);line([(1663,313),(1689,313)]);txt(1728,264,'p₀(Q)',12)
# point evidence bus: correct source is projected features and support clicks.
box(1790,279,120,75,'Point evidence\nfrom Fₛ, Fq\nand S clicks',9,fc='#f5faf7');line([(1790,316),(1776,316),(1776,367),(1615,367),(1615,355)],name='point_evidence_to_correction')
# LOWER CENTER: support decoder, feature correspondence, anchor and transport.
box(321,684,125,69,'Shared decoder\n+ point correction\non support S',8.5)
txt(383,677,'Rₛ + q',8);line([(382,665),(382,671)],name='support_decoder_inputs')
# task continuation via named ports is deliberate to avoid crossing full figure.
txt(1810,240,'q also conditions support decoding',8,color='#2c7554')
heat(479,689,64,59,2);line([(447,718),(478,718)]);txt(523,678,'pₛ (detach)',8)
box(570,686,106,63,'Click anchors\n+ / −',10,fc='#f9fcf7');line([(544,718),(569,718)]);txt(623,768,'S clicks',9);line([(623,757),(623,751)],name='support_click_anchors')
heat(705,691,60,57,2);line([(677,718),(704,718)]);txt(735,675,'p̃ₛ',12)
# Feature matrices upper row in transfer panel.
cube(325,606,35,24,'#9ab7dd');cube(325,649,35,24,'#b5a4d5');txt(381,614,'Fₛ',10);txt(381,656,'Fq',10)
line([(392,615),(411,615),(411,639),(447,639)]);line([(392,655),(411,655),(411,639)],arrow=False)
heat(451,613,65,52,1,'aff');txt(484,603,'C₁₆',10);txt(483,674,'16×16 grid',7)
heat(655,613,65,52,3,'aff');txt(687,603,'C₃₂',10);txt(688,679,'32×32 grid',8)
line([(517,640),(654,640)]);txt(585,624,'Coarse prior',9)
# Fine affinity also consumes projected image features, not just coarse affinity.
line([(411,615),(411,595),(687,595),(687,612)],color='#6c779a',name='fine_similarity_features');txt(558,588,'Fine S/Q similarity',8,color='#6c779a')
plus(830,715,15,'×');line([(766,718),(814,718)]);line([(721,640),(830,640),(830,698)])
heat(887,685,76,64,5);line([(846,715),(886,715)]);txt(924,670,'Fine transfer',9)
box(1000,684,163,65,'Coarse transfer\n+ matching certainty',9,fc='#fcf4e8')
line([(465,666),(465,787),(1082,787),(1082,750)],name='coarse_match_to_transport')
line([(735,749),(735,775),(1017,775),(1017,750)],color='#947858',name='support_prob_to_coarse_transport');line([(720,650),(979,650),(979,706),(999,706)],name='fine_affinity_certainty')
# RIGHT LOWER: no accidental route to logit from residual input.
# p0 -> logit -> summation, and separate p0 -> residual head.
line([(1728,354),(1728,408),(1305,408),(1305,478)],name='query_probability_to_logit')
box(1265,479,80,43,'logit',11);line([(1346,500),(1580,500),(1580,574)],name='initial_logits_to_sum');txt(1484,481,'ℓ₀(Q)',12)
box(1255,654,126,74,'Residual head\nh (8 channels)',10,fc='#fcf3e7')
line([(963,714),(1206,714),(1206,679),(1254,679)],color='#b97531',name='fine_transfer_to_residual')
line([(1164,717),(1218,717),(1218,708),(1254,708)],color='#b97531',name='coarse_certainty_to_residual')
line([(1728,408),(1728,432),(1237,432),(1237,646),(1318,646),(1318,653)],color='#397c63',name='p0_uncertainty_to_residual');txt(1411,449,'p₀ and 4p₀(1−p₀)',8,color='#397c63')
line([(1850,355),(1850,456),(1402,456),(1402,637),(1354,637),(1354,653)],color='#658376',name='signed_point_evidence_to_residual')
box(1410,667,85,43,'2 tanh',11);line([(1382,691),(1409,691)]);heat(1520,655,67,66,2,'res');line([(1496,688),(1519,688)]);txt(1553,742,'Δℓ(Q)',12)
txt(1492,624,'Upsample',8);plus(1580,591,16);line([(1554,654),(1554,625),(1580,625),(1580,608)],name='residual_to_sum')
box(1635,569,85,45,'sigmoid',10);line([(1597,591),(1634,591)])
box(1761,570,71,43,'≥ 0.5',10);line([(1721,591),(1760,591)])
# Binary mask vector silhouette and illustration thumbnail, explicitly not evidence.
rect(1851,512,69,71,fc='#121719',ec='#6e7c84');poly=np.array([[.12,.75],[.25,.5],[.2,.31],[.4,.2],[.49,.08],[.65,.18],[.83,.34],[.76,.54],[.85,.66],[.58,.83],[.33,.79],[.23,.94]])*[69,71]+[1851,512];ax.add_patch(ident(Polygon(poly,fc='white',ec='none'),'binary_mask'))
im(imgs[5],1840,655,80,80);line([(1833,591),(1841,591),(1841,547),(1850,547)]);line([(1841,591),(1841,627),(1880,627),(1880,654)])
txt(1884,495,'Binary mask',9);txt(1881,754,'H&E overlay',9)
txt(1560,551,'p(Q) = sigmoid(ℓ₀(Q) + Δℓ(Q))',12,bold=True)
txt(1553,787,'Internal maps and WSI site markers are schematic; histology is real.',9)
# export, validate text editing and embedded assets
for ext in ['svg','pdf','png']:fig.savefig(O/f'FewClick_method_corrected.{ext}',dpi=170,facecolor='white')
svg=O/'FewClick_method_corrected.svg';root=ET.parse(svg).getroot();ns={'s':'http://www.w3.org/2000/svg'}
summary={'svg_text_elements':len(root.findall('.//s:text',ns)),'svg_images':len(root.findall('.//s:image',ns)),'svg_paths':len(root.findall('.//s:path',ns)),'all_images_embedded':all(i.attrib.get('{http://www.w3.org/1999/xlink}href','').startswith('data:') for i in root.findall('.//s:image',ns)),'source_preview_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert summary['svg_text_elements']>100 and summary['all_images_embedded'];(O/'validation.json').write_text(json.dumps(summary,indent=2));print(summary)
