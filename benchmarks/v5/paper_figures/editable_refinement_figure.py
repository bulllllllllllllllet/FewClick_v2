"""Faithful editable reconstruction: native text/wires/containers and separate visual assets."""
from pathlib import Path
import io,json,base64,html,hashlib,zipfile
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement
SRC=Path('论文/CVPR_HG_mix/figures/method_final_generated_1115/vector_frames/FewClick_method_vector_frames.png')
OUT=SRC.parent.parent/'editable_refinement_1115';OUT.mkdir(exist_ok=True);assets=OUT/'assets';assets.mkdir(exist_ok=True)
im=Image.open(SRC).convert('RGB');W,H=im.size
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(16*H/W);slide=prs.slides.add_slide(prs.slide_layouts[6]);slide.shapes.turbo_add_enabled=True
u=16*914400/W;pf=16*72/W
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="context-stroke"/></marker><linearGradient id="blue" x2="0" y2="1"><stop stop-color="#e9f8ff"/><stop offset="1" stop-color="#bce7fa"/></linearGradient><linearGradient id="green" x2="0" y2="1"><stop stop-color="#edf8e5"/><stop offset="1" stop-color="#c5e6b6"/></linearGradient><linearGradient id="orange" x2="0" y2="1"><stop stop-color="#fff7e9"/><stop offset="1" stop-color="#ffe3b4"/></linearGradient><linearGradient id="peach" x2="0" y2="1"><stop stop-color="#fff4eb"/><stop offset="1" stop-color="#ffd0b8"/></linearGradient></defs>']
manifest=[];counts={'text':0,'wires':0,'shapes':0,'pictures':0};counter=0
INK='#101010'
def eid(kind,name):
 global counter
 counter+=1;return f'{kind}-{counter:03d}-{name}'
def rgb(s):return RGBColor.from_string(s.lstrip('#'))
def style(sh,fill,stroke,lw):
 if fill:sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
 else:sh.fill.background()
 if stroke:sh.line.color.rgb=rgb(stroke);sh.line.width=Pt(lw*pf)
 else:sh.line.fill.background()
 sh._element.spPr.append(OxmlElement('a:effectLst'))
def rect(x,y,w,h,fill='white',stroke=INK,lw=1.4,r=0,name='box',gradient=None):
 ident=eid('shape',name);fc=fill or 'none';svg.append(f'<rect id="{ident}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{("url(#"+gradient+")") if gradient else fc}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
 sh=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE,round(x*u),round(y*u),round(w*u),round(h*u));sh.name=ident
 if r:
  try:sh.adjustments[0]=min(.5,r/min(w,h))
  except Exception:pass
 style(sh,fill,stroke,lw);counts['shapes']+=1
 return sh
def text(x,y,s,size=18,bold=False,color=INK,ha='center',name='label',italic=False):
 ident=eid('text',name);lines=s.split('\n');spacing=size*1.12;first=y-(len(lines)-1)*spacing/2
 svg.append(f'<text id="{ident}" x="{x}" y="{first}" dominant-baseline="central" text-anchor="{ {"center":"middle","left":"start","right":"end"}[ha]}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}" font-style="{"italic" if italic else "normal"}" fill="{color}">'+''.join(f'<tspan x="{x}" dy="{0 if i==0 else spacing}">{html.escape(t)}</tspan>' for i,t in enumerate(lines))+'</text>')
 # Use generous text boxes, with fixed explicit type sizes (no autofit).
 width=max(size*len(t)*.64 for t in lines)+12;height=spacing*len(lines)+4
 left=x-width/2 if ha=='center' else x if ha=='left' else x-width
 sh=slide.shapes.add_textbox(round(left*u),round((y-height/2)*u),round(width*u),round(height*u));sh.name=ident;tf=sh.text_frame;tf.clear();tf.word_wrap=False;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for i,t in enumerate(lines):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=t;p.alignment={'center':PP_ALIGN.CENTER,'left':PP_ALIGN.LEFT,'right':PP_ALIGN.RIGHT}[ha];p.space_before=p.space_after=Pt(0);p.line_spacing=1.0
  for run in p.runs:run.font.name='Arial';run.font.size=Pt(size*pf);run.font.bold=bold;run.font.italic=italic;run.font.color.rgb=rgb(color)
 counts['text']+=1
 return sh
def wire(pts,color=INK,lw=1.8,arrow=True,dashed=False,name='connector'):
 ident=eid('wire',name);svg.append(f'<polyline id="{ident}" points="'+ ' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="miter"'+(' stroke-dasharray="5 3"' if dashed else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
 b=slide.shapes.build_freeform(pts[0][0]*u,pts[0][1]*u,scale=1);b.add_line_segments([(x*u,y*u) for x,y in pts[1:]],close=False);sh=b.convert_to_shape();sh.name=ident;style(sh,None,color,lw)
 if arrow:
  ar=OxmlElement('a:tailEnd');ar.set('type','triangle');ar.set('w','sm');ar.set('len','sm');sh._element.spPr.get_or_add_ln().append(ar)
 if dashed:
  d=OxmlElement('a:prstDash');d.set('val','dash');sh._element.spPr.get_or_add_ln().append(d)
 counts['wires']+=1
 return sh
def picture(bounds,name,dest=None):
 tile=im.crop(bounds);p=assets/(name+'.png');tile.save(p);x,y,x1,y1=bounds
 if dest:x,y,x1,y1=dest
 data=base64.b64encode(p.read_bytes()).decode();ident=eid('image',name);svg.append(f'<image id="{ident}" x="{x}" y="{y}" width="{x1-x}" height="{y1-y}" xlink:href="data:image/png;base64,{data}"/>')
 sh=slide.shapes.add_picture(str(p),round(x*u),round(y*u),round((x1-x)*u),round((y1-y)*u));sh.name=ident
 manifest.append({'id':ident,'file':str(p),'source_crop':bounds,'placement':[x,y,x1,y1]});counts['pictures']+=1
 return sh
def circle(x,y,r=16,label='+',color=INK):
 ident=eid('shape','operator');svg.append(f'<circle id="{ident}" cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="1.8"/>');sh=slide.shapes.add_shape(MSO_SHAPE.OVAL,round((x-r)*u),round((y-r)*u),round(2*r*u),round(2*r*u));sh.name=ident;style(sh,'#ffffff',color,1.8);text(x,y,label,24,color=color);counts['shapes']+=1

def box(x,y,w,h,s,size=18,fill='#ffffff',name='module'):
 rect(x,y,w,h,fill,INK,1.5,5,name);text(x+w/2,y+h/2,s,size,name=name+'-text')
def encoder(x,y,w,h,label='Phikon-v2\n(shared)\nE'):
 ident=eid('shape','encoder');pts=[(x,y),(x+w,y+12),(x+w,y+h-12),(x,y+h)];svg.append(f'<polygon id="{ident}" points="'+ ' '.join(f'{a},{b}' for a,b in pts)+'" fill="#f7f7f7" stroke="#111" stroke-width="1.7"/>');b=slide.shapes.build_freeform(x*u,y*u,scale=1);b.add_line_segments([(a*u,b*u) for a,b in pts[1:]],close=True);sh=b.convert_to_shape();sh.name=ident;style(sh,'#f7f7f7',INK,1.7);text(x+w*.48,y+h*.50,label,18);counts['shapes']+=1

def tokens(x,y,w=118,color='#b6d7e4',name='region-tokens'):
 for k in range(3):
  for j in range(9):rect(x+j*w/9,y+k*25,w/9,11,color,'#5a9ba4',.45,name=name)
  text(x+w+24,y+k*25+6,['R1','R2','R64'][k],16)
 text(x+8,y+43,'⋮',18)

def panel(x,y,w,h,title,color,fill,grad,header=44):
 rect(x,y,w,h,'#ffffff',color,2.5,10,'panel');rect(x+4,y+4,w-8,header-4,fill,None,0,0,'header',grad);wire([(x+3,y+header),(x+w-3,y+header)],color,1.2,False);text(x+w/2,y+header/2,title,26,True,color)
# Four exact panel boxes.
panel(7,9,313,791,'Multiscale Inputs','#071f9c','#c8eafa','blue')
# Border colors are separate from dark title typography.
panel(328,9,876,529,'Visual Encoding & Hierarchical Regions','#076c28','#d1e9c0','green')
rect(329,546,875,255,'#fff8ef','#ff9200',2.5,10,'transfer-panel');rect(333,550,867,26,'#ffe9c5',None,0,0,'transfer-header','orange');text(766,563,'Coarse-to-Fine Transfer',25,True,'#b94a00')
panel(1213,9,727,792,'Support-conditioned Prediction','#e64c00','#ffdac5','peach')
# Explicitly match frame colors, rather than title colors.
rect(7,9,313,791,None,'#08afe4',2.5,10,'outer-blue');rect(328,9,876,529,None,'#0aaa15',2.5,10,'outer-green')
# Left image stack: preserve histology and embedded support/query marks as image texture.
for b,n in [((114,76,271,200),'input-fine'),((114,230,271,341),'input-context2'),((114,368,271,482),'input-context4'),((23,518,310,680),'same-slide-wsi')]:picture(b,n)
for y,s in [(150,'Fine\n1×'),(283,'Context\n2×'),(438,'Context\n4×')]:text(65,y,s,22,True)
for p in [[(114,201),(164,268)],[(271,201),(212,268)],[(114,342),(164,408)],[(271,342),(212,408)],[(114,483),(104,548)],[(271,483),(140,548)]]:wire(p,'#222222',1.1,False,True)
text(164,706,'Same-slide support S /\nquery Q',18)
for y,sgn,col,lab in [(745,'+','#009a19','Positive click (S only)'),(775,'−','#f01b21','Negative click (S only)')]:
 circle(34,y,12,sgn,col);text(180,y,lab,18)
# Fine inset.
rect(344,57,842,163,'#ffffff','#222222',1.2,10,'fine-inset');text(362,72,'Fine (1×): whole + 4 tiles',20,True,ha='left')
picture((375,102,472,181),'fine-whole');picture((518,101,607,180),'fine-four-tiles');text(424,193,'whole tile',18);text(564,198,'4 tiles\n(2×2)',17)
encoder(655,94,120,103)
text(755,94,'❄',24,color='#009be8')
wire([(470,102),(470,90),(634,90),(634,117),(655,117)]);wire([(609,166),(654,166)])
picture((812,72,894,136),'whole-feature-cube');picture((813,143,895,209),'tiled-feature-cube')
wire([(776,143),(789,143),(789,111),(810,111)]);wire([(789,143),(789,175),(811,175)])
box(912,87,47,27,'0.5',17);box(912,166,47,27,'0.5',17);wire([(895,100),(910,100)],arrow=False);wire([(959,100),(984,100),(984,122)]);wire([(895,179),(910,179)],arrow=False);wire([(959,179),(984,179),(984,155)])
circle(984,138,16);wire([(1001,138),(1048,138)]);picture((1048,91,1151,162),'fused-feature-cube');text(1102,186,'Fine feature\n(fused)',19,True)
# Region rows.
wire([(1111,207),(1111,235),(508,235),(508,266)],'#8208b0',2.2)
for k,y in enumerate([290,367,450]):
 text(376,y,['Fine\n(1×)','Context\n(2×)','Context\n(4×)'][k],19,True)
 if k==0:box(414,268,132,47,'(from fused\nfine feature)',17,'#efd9fa','fused-input')
 else:encoder(422,y-35,106,70);text(511,y-43,'❄',23,color='#009be8')
 wire([(548 if k==0 else 529,y),(558,y)])
 picture([(565,263,621,319),(558,340,622,402),(558,419,624,483)][k],f'region-feature-cube-{k}')
 wire([(623,y),(639,y)]);box(640,y-20,90,43,'A: soft\nassignment',17,name='soft-assignment')
 wire([(731,y),(744,y)]);picture([(746,260,811,322),(746,337,811,401),(746,416,811,480)][k],f'soft-regions-{k}');wire([(812,y),(844,y)])
 rect(847,y-25,183,72,'#e0f1f6','#5da4b8',1,10,'region-token-bank');tokens(862,y-18,105,['#b8dce1','#9fcedf','#87b8d8'][k])
 wire([(1034,y),(1048,y),(1048,367),(1066,367)])
text(951,251,'Region Token (64 regions / scale)',18,True)
rect(1068,263,123,230,'#ffffff','#222222',1.1,10,'cross-scale-panel');text(1128,283,'Cross-scale\ncontext',18,True)
picture((1081,309,1183,384),'cross-scale-tree');picture((1082,392,1129,482),'context-token-strips')
for y,s in [(402,'R1'),(431,'…'),(472,'R64')]:text(1160,y,s,18,True)
wire([(400,496),(400,513),(638,513)],arrow=False);wire([(896,513),(1136,513),(1136,498)],arrow=False);box(638,499,257,27,'Shared S / Q encoding',19)
# Input scale connectors.
for yy,y,col in [(136,289,'#ed1111'),(295,371,'#004dff'),(453,453,'#ff8d00')]:wire([(272,yy),(311,yy),(311,y),(347,y)],col,1.7,True,True)
# Support prompt strip. Crops retain the exact internal microscopy and region polygons.
text(1337,72,'Support S + clicks',19,True);picture((1246,94,1408,215),'support-with-clicks');wire([(1410,150),(1454,150)]);picture((1458,93,1588,214),'support-region-map');text(1522,72,'Region map (S)',19,True)
wire([(1589,151),(1605,151),(1605,125),(1620,125)]);wire([(1605,151),(1605,177),(1620,177)])
for y,col,lab in [(110,'#acdcb7','q⁺'),(164,'#eeaaaa','q⁻')]:
 rect(1622,y,121,30,'#ffffff','#a4c4af',1,4,'signed-prompt-token')
 for j in range(10):rect(1628+j*8,y+6,8,16,col,'#639b78',.5,name='prompt-token-cell')
 text(1725,y+15,lab,18,True)
wire([(1745,125),(1761,125),(1761,149),(1772,149)]);wire([(1745,179),(1761,179),(1761,153),(1772,153)])
box(1772,134,48,34,'MLP',18);wire([(1821,151),(1844,151)])
for j in range(8):rect(1845+j*10,143,10,14,'#dddddd','#777777',.6,name='task-token')
text(1886,119,'Task q',20,True)
wire([(1796,169),(1796,236),(1429,236),(1429,319)])
# Query route and native probability strips.
wire([(1191,366),(1253,366)]);box(1191,334,64,28,'Query R',16)
text(1292,280,'Region Query\nToken\n(from center)',18,True)
rect(1257,317,89,103,'#ffffff','#67a1b0',1,6,'query-tokens')
for y in [323,347,394]:
 rect(1263,y,50,15,'#b5dbe5','#608591',.6,name='query-strip');text(1326,y+8,'R1' if y==323 else 'R2' if y==347 else 'R64',16,True)
text(1283,377,'⋮',18);wire([(1346,368),(1361,368)])
box(1362,323,132,87,'Context-aware\nRegion\nDecoder',18,'#d6ecf8','query-decoder');wire([(1495,368),(1511,368)])
text(1560,279,'Region\nProbability\nR',19,True)
rect(1514,318,95,101,'#ffffff','#777777',1,4,'region-probabilities')
for y in [322,347,395]:
 picture((1084,394,1127,407),'prob-strip-'+str(y),(1518,y,1563,y+16));text(1585,y+8,'R1' if y==322 else 'R2' if y==347 else 'R64',17,True)
text(1583,377,'…',18);wire([(1610,368),(1629,368)])
box(1630,323,87,86,'Soft\nRegion-to-\nPixel\nProjection',17,name='projection');wire([(1718,368),(1734,368)])
box(1736,324,88,85,'Point\nCorrection',19,name='point-correction');wire([(1825,367),(1844,367)])
picture((1847,315,1932,414),'query-initial-prob');text(1886,291,'p₀(Q)',24,italic=True)
# Coarse-to-fine feature affinities.
picture((406,579,464,626),'support-projected-feature');picture((407,630,466,676),'query-projected-feature');text(383,601,'Fₛ',22,italic=True);text(384,650,'Fq',22,italic=True)
wire([(466,599),(495,599),(495,632),(516,632)],arrow=False);wire([(467,649),(495,649),(495,632)],arrow=False)
wire([(516,632),(539,632),(539,589),(847,589),(847,604)]);wire([(624,589),(624,605)])
text(624,617,'C₁₆ (16×16)',20,True);picture((590,632,670,679),'affinity-coarse')
text(848,617,'C₃₂ (32×32)',20,True);picture((809,631,888,679),'affinity-fine')
wire([(671,659),(806,659)]);text(740,639,'Coarse\nprior',18,True)
box(362,691,105,72,'Support\ndecoder\n(Rₛ, q)',18,name='support-decoder');wire([(468,728),(519,728)]);picture((522,707,606,753),'support-probability');text(565,687,'pₛ\n(detach)',17,True)
wire([(608,727),(638,727)]);box(640,702,114,52,'Click anchors\n+ / −',18,name='click-anchors');wire([(755,727),(794,727)])
picture((797,703,882,754),'anchored-support-prob');text(806,688,'p̃ₛ',22,italic=True)
wire([(884,726),(911,726),(911,664),(957,664)]);wire([(890,637),(975,637),(975,648)]);circle(975,664,16,'×');wire([(992,664),(1031,664)])
picture((1034,634,1117,689),'fine-transferred-prob');text(1075,610,'Fine\ntransfer',20,True)
rect(926,721,226,53,'#ffffff',INK,1.6,6,'coarse-transfer');text(986,747,'Coarse transfer\n+ certainty',16,True);picture((1058,728,1135,768),'coarse-transfer-map')
wire([(883,727),(900,727),(900,751),(923,751)]);wire([(629,681),(629,776),(923,776),(923,759)])
# Right lower: visual refinement schematic; the rest of the figure is unchanged.
wire([(1888,416),(1888,445),(1264,445),(1264,547)],arrow=False)
text(1518,488,'Prediction Refinement',23,True)
box(1227,548,157,48,'Initial prediction\ncues',18,name='initial-prediction-cues')
# Initial prediction bypass, converted to logits at the named fusion operation.
wire([(1385,572),(1400,572),(1400,535),(1574,535),(1574,579)])
text(1488,520,'Initial prediction',18)
wire([(1300,597),(1300,640)])
box(1230,642,195,72,'Bounded residual\nrefinement',19,name='bounded-residual-refinement')
wire([(1119,664),(1170,664),(1170,678),(1227,678)]);wire([(1154,746),(1173,746),(1173,710),(1227,710)])
box(1247,728,122,61,'Point cues\n(from S / Q)\n+ / −',18,name='point-cues');wire([(1300,727),(1300,715)]);wire([(1370,751),(1782,751),(1782,412)])
wire([(1426,679),(1464,679)])
picture((1465,652,1542,709),'bounded-logit-residual');text(1500,634,'Correction',18)
wire([(1543,680),(1574,680),(1574,617)]);circle(1574,598,17)
text(1501,598,'Logit fusion',18,True)
wire([(1592,598),(1618,598)]);box(1620,574,155,48,'Mask\nprediction',19,name='mask-prediction')
wire([(1777,599),(1800,599),(1800,544),(1820,544)]);wire([(1800,599),(1800,698),(1820,698)])
picture((1823,507,1931,590),'binary-mask');text(1875,482,'Binary mask',20,True)
picture((1822,652,1932,747),'prediction-overlay');text(1873,624,'H&E overlay',20,True)
text(1576,779,'Method schematic; internal maps are illustrative.',18,italic=True)
svg.append('</svg>');(OUT/'FewClick_分层可编辑.svg').write_text('\n'.join(svg))
slide.notes_slide.notes_text_frame.text='Faithful reconstruction from the approved generated illustration. Native editable text, arrows and containers. Histology, heatmaps, region polygons, textured cubes and tree are separate raster components. They are schematic generated imagery, not measured experimental results. The SVG has the same separately addressable objects.'
ppt=OUT/'FewClick_分层可编辑.pptx';prs.save(ppt)
# Remove inherited shadows consistently across renderers.
from lxml import etree
with zipfile.ZipFile(ppt) as z:payload={n:z.read(n) for n in z.namelist()}
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for n,data in list(payload.items()):
 if n.endswith('.xml') and (n.startswith('ppt/slides/') or n.startswith('ppt/theme/')):
  root=etree.fromstring(data)
  for el in root.xpath('//a:effectRef',namespaces=ns):el.set('idx','0')
  for el in root.xpath('//a:effectLst',namespaces=ns):el.clear()
  payload[n]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(ppt,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in payload.items():z.writestr(n,data)
check=Presentation(ppt);assert len(check.slides)==1
(OUT/'组件索引.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/'validation.json').write_text(json.dumps({'counts':counts,'source':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'pptx_sha256':hashlib.sha256(ppt.read_bytes()).hexdigest(),'svg_text_elements':sum('<text ' in s for s in svg),'svg_arrow_elements':sum('<polyline ' in s for s in svg)},indent=2))
(OUT/'使用说明.md').write_text('以已确认的边框修正版为定位参考，保持1946×808比例。\n\nSVG和PPT中文字、箭头、容器、外框、token格子为独立原生元素。病理图、热图、区域图、有纹理特征块和层级树为独立图片组件，可分别移动、缩放或替换；它们内部的纹理像素不是矢量。\n\n推荐用PPT编辑文字和箭头，或用Inkscape/Illustrator打开SVG；每个对象有语义名称。assets目录保留全部独立图片，组件索引.json记录来源坐标。图内图片沿用生成式机制示意，不作为实验结果。\n')
print(OUT);print(counts)
