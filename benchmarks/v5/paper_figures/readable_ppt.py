"""Translate the verified matplotlib diagram into native PowerPoint shapes."""
from pathlib import Path
import io,json,datetime,hashlib
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
src=Path('benchmarks/v5/paper_figures/method_vector.py');source=src.read_text()
# Extract the existing embedded bitmaps; do not reopen WSI/NFS sources.
x=source.index('# Use real tissue inserts');y=source.index('plt.rcParams.update',x)
local_images="""import base64,xml.etree.ElementTree as ET,io
svg=ET.parse('论文/CVPR_HG_mix/figures/method_v5_vector_1114/FewClick_method_corrected.svg').getroot()
images=[np.asarray(Image.open(io.BytesIO(base64.b64decode(v.attrib['{http://www.w3.org/1999/xlink}href'].split(',')[1]))).convert('RGB')) for v in svg.findall('.//{http://www.w3.org/2000/svg}image')]
imgs=[images[i] for i in [0,1,2,3,9,10]]
"""
source=source[:x]+local_images+source[y:]
a=source.index('panel(6,7,');b=source.index('# export, validate')
ctx={'__name__':'editable_ppt','__file__':str(src)};exec(compile(source[:a],str(src),'exec'),ctx)
records=[];depth=[0]
def wrap(name,fn):
 def f(*args,**kwargs):
  top=depth[0]==0;before=set(ctx['ax'].get_children()) if top else None;depth[0]+=1
  try:return fn(*args,**kwargs)
  finally:
   depth[0]-=1
   if top:records.append((name,args,[x for x in ctx['ax'].get_children() if x not in before]))
 return f
for name in ['cube','tokens','regions','enc','heat','im','box','plus','click']:ctx[name]=wrap(name,ctx[name])
exec(compile(source[a:b],str(src),'exec'),ctx)
fig,ax=ctx['fig'],ctx['ax']
relabels={
 '2  Visual Encoding & Hierarchical Regions':'2  Encoding & Hierarchical Regions',
 '3  Support-conditioned Prediction':'3  Support-conditioned Prediction',
 'Fine view: whole + tiled encoding':'Fine view: whole + tiles',
 'Phikon-v2: shared / frozen':'Shared frozen Phikon-v2',
 'Phikon-v2\nshared / frozen':'Phikon-v2',
 'E: whole':'Whole', 'E: each tile':'Tiles',
 'Fused fine\nfeature grid':'Fused fine\nfeatures',
 'Fused features\nfrom inset':'Fused fine',
 'Region tokens (64 / scale)':'Region tokens',
 'Cross-scale context':'Cross-scale\ncontext',
 'Shared support / query encoding • soft assignment A • hierarchical region features':'Shared S/Q encoding    •    A: soft assignment',
 'Same-slide support S / query Q':'Same-slide S / Q',
 'Positive + / negative − clicks on S':'Support clicks: + / −',
 'Aligned physical fields of view':'Aligned multiscale views',
 'Prompt → region tokens':'Prompt → regions',
 'Context-aware\nregion decoder':'Region\ndecoder',
 'Soft region\n→ pixel\nprojection':'Region-to-\npixel',
 'Signed point\nsimilarity\ncorrection':'Point\ncorrection',
 'Point evidence\nfrom Fₛ, Fq\nand S clicks':'Point cues\nFₛ, Fq, clicks',
 'Shared decoder\n+ point correction\non support S':'Support\ndecoder',
 'q also conditions support decoding':'Shared task q for S / Q',
 'Coarse transfer\n+ matching certainty':'Coarse transfer\n+ certainty',
 'Residual head\nh (8 channels)':'Residual head\n8 channels',
 'Internal maps and WSI site markers are schematic; histology is real.':'Internal maps are schematic.',
 'Fine S/Q similarity':'Fine S/Q match',
}
for ar in ax.get_children():
 if isinstance(ar,Text):
  old=ar.get_text();ar.set_text(relabels.get(old,old))
  # Scale labels relative to graphics, not merely the physical slide size.
  fs=ar.get_fontsize();ar.set_fontsize(min(23,max(13.5,fs*1.55)))
  if old=='Cross-scale context':ar.set_fontsize(13)
  if old=='Fine S/Q similarity':ar.set_position((558,600));ar.set_fontsize(12)
  if old in ('16×16 grid','32×32 grid'):ar.set_text('')
  if old=='pₛ (detach)':ar.set_position((511,677));ar.set_text('pₛ')
  if old=='Rₛ + q':ar.set_position((381,670))
  if old=='Soft regions':ar.set_fontsize(13)
  if old in ('Align','Stitch') or (old=='E' and ar.get_position()[1]<210):ar.set_text('')
  if old=='Support S + clicks':ar.set_text('Support clicks')
  if old=='Prompt → region tokens':ar.set_text('Prompt regions')
  if old=='Fused fine\nfeature grid':ar.set_position((1090,147))
fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.canvas.get_width_height();dpi=fig.dpi;scale=914400/dpi
out=Path('论文/CVPR_HG_mix/figures')/('readable_ppt_'+datetime.datetime.now().strftime('%m%d_%H%M%S'));out.mkdir()
prs=Presentation();prs.slide_width=Inches(24);prs.slide_height=Inches(24*826/1944);slide=prs.slides.add_slide(prs.slide_layouts[6]);shape_map={};stats={'text':0,'pictures':0,'vector_shapes':0,'groups':0}
# Remove vertical whitespace while retaining the same three-column structure.
y_old=np.array([0,50,210,240,491,552,595,815,826]);y_new=y_old.copy()
unit=24/1944; font_factor=1.0
inv=ax.transData.inverted()
def position(px,py):
 x,y=inv.transform((px,py));return x*unit*914400,float(np.interp(y,y_old,y_new))*unit*914400
def bounds(bb):
 x0,y0=position(bb.x0,bb.y1);x1,y1=position(bb.x1,bb.y0);return tuple(round(v) for v in (x0,y0,x1-x0,y1-y0))
def emu(x):return int(round(x*scale))
def rgba(color):return tuple(float(v) for v in color)
def style(sh,face,edge,width):
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 if face is None or len(face)==4 and face[3]==0:sh.fill.background()
 else:
  sh.fill.solid();sh.fill.fore_color.rgb=RGBColor(*(int(round(max(0,min(1,c))*255)) for c in face[:3]))
  if len(face)==4 and face[3]<1:
   n=OxmlElement('a:alpha');n.set('val',str(round(face[3]*100000)));sh.fill._xPr.solidFill.srgbClr.append(n)
 if edge is None or len(edge)==4 and edge[3]==0:sh.line.fill.background()
 else:sh.line.color.rgb=RGBColor(*(round(c*255) for c in edge[:3]));sh.line.width=Pt(max(.2,width*font_factor))
def vector(artist,path,face,edge,width,closed):
 polys=path.to_polygons(closed_only=False)
 if not polys:return []
 builder=None
 for poly in polys:
  if len(poly)<2:continue
  xy=[position(float(p[0]),float(p[1])) for p in poly]
  if builder is None:builder=slide.shapes.build_freeform(xy[0][0],xy[0][1],scale=1)
  else:builder.move_to(*xy[0])
  builder.add_line_segments(xy[1:],close=closed)
 if builder is None:return []
 sh=builder.convert_to_shape();sh.name=artist.get_gid() or 'vector';style(sh,face,edge,width);stats['vector_shapes']+=1;return [sh]
slide.shapes.turbo_add_enabled=True
for artist in sorted([x for x in ax.get_children() if x.get_gid()],key=lambda x:x.get_zorder()):
 shapes=[]
 if isinstance(artist,Text):
  bb=artist.get_window_extent(renderer);pad=3;from matplotlib.transforms import Bbox
  bb=Bbox.from_extents(bb.x0-pad,bb.y0-5,bb.x1+pad+4,bb.y1+2);sh=slide.shapes.add_textbox(*bounds(bb));sh.name=artist.get_gid();tf=sh.text_frame;tf.clear();tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.TOP
  for i,line in enumerate(artist.get_text().split('\n')):
   p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.alignment=PP_ALIGN.CENTER if artist.get_ha()=='center' else PP_ALIGN.RIGHT if artist.get_ha()=='right' else PP_ALIGN.LEFT;p.space_after=Pt(0);p.space_before=Pt(0);p.line_spacing=1.05
   for r in p.runs:r.font.name='Arial';r.font.size=Pt(max(2.7,artist.get_fontsize()*font_factor));r.font.bold=artist.get_fontweight() in ['bold',700];r.font.color.rgb=RGBColor(*(round(c*255) for c in __import__('matplotlib').colors.to_rgba(artist.get_color())[:3]))
  shapes=[sh];stats['text']+=1
 elif isinstance(artist,AxesImage):
  bb=artist.get_window_extent(renderer);arr=np.asarray(artist.get_array());buf=io.BytesIO();Image.fromarray(arr.astype('uint8')).save(buf,format='PNG');buf.seek(0);sh=slide.shapes.add_picture(buf,*bounds(bb));sh.name=artist.get_gid();shapes=[sh];stats['pictures']+=1
 elif isinstance(artist,Patch):
  path=artist.get_path().transformed(artist.get_transform());shapes=vector(artist,path,artist.get_facecolor(),artist.get_edgecolor(),artist.get_linewidth(),artist.get_fill())
 elif isinstance(artist,Line2D):
  path=artist.get_path().transformed(artist.get_transform());edge=__import__('matplotlib').colors.to_rgba(artist.get_color());shapes=vector(artist,path,None,edge,artist.get_linewidth(),False)
 shape_map[artist]=shapes
# Components become editable groups. The slide's main backgrounds remain independently selectable.
for kind,args,artists in records:
 shapes=[sh for ar in artists for sh in shape_map.get(ar,[])]
 if len(shapes)>1:
  parent=slide.shapes._spTree;indices=[parent.index(sh._element) for sh in shapes];last=max(indices);following=parent[last+1] if last+1<len(parent) else None
  g=slide.shapes.add_group_shape(shapes)
  if following is not None and following.getparent() is parent:parent.remove(g._element);parent.insert(parent.index(following),g._element)
  g.name=f'{kind}_{stats["groups"]+1:03d}';stats['groups']+=1
slide.notes_slide.notes_text_frame.text='FewClick HG-mix method diagram. Native editable text and vectors; histology inserts are independent images. Internal heatmaps, soft regions and WSI markers are schematic. Preserve the computational ordering: support-conditioned region+point prediction p0; click-anchored support probability transfer; bounded residual added to logits before sigmoid. All elements grouped by component. Un-group or enter a group to edit internal shapes.'
ppt=out/'FewClick_AAAI比例大字版_可编辑.pptx';prs.save(ppt)
# Suppress theme effects explicitly for consistent PowerPoint/LibreOffice rendering.
import zipfile
from lxml import etree
with zipfile.ZipFile(ppt) as archive:
 payload={name:archive.read(name) for name in archive.namelist()}
ns_a={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for name, data in list(payload.items()):
 if name.endswith('.xml') and (name.startswith('ppt/slides/') or name.startswith('ppt/theme/')):
  root=etree.fromstring(data)
  for effect in root.xpath('//a:effectRef',namespaces=ns_a):effect.set('idx','0')
  for effect in root.xpath('//a:effectLst',namespaces=ns_a):effect.clear()
  payload[name]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile(ppt,'w',zipfile.ZIP_DEFLATED) as archive:
 for name,data in payload.items():archive.writestr(name,data)
# Round-trip parse verifies shape tree and exactly one slide.
check=Presentation(ppt);assert len(check.slides)==1
(out/'validation.json').write_text(json.dumps(dict(stats,slides=1,source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),pptx_sha256=hashlib.sha256(ppt.read_bytes()).hexdigest()),indent=2))
(out/'使用说明.md').write_text('# 单页可编辑方法图\n\n打开PPTX即可编辑；不是整页图片。文字为原生文本框，箭头/区域/特征立方体/token/热图为原生矢量形状，病理图为独立图片。完整组件已组合，可整体移动；需要改细节时进入组或取消组合。正文图1未替换。\n\n沿用已批准的蓝绿橙三栏及校正后的连接；字体使用Arial便于跨机器编辑。画布恢复24英寸宽、约10.20英寸高，与原AAAI/生成图的长宽比接近。文字相对图形放大约55%，常规标签至少13.5pt；缩短重复说明，保留双路径、粗细迁移和残差融合。插入论文时整体按正文通栏宽度缩放。\n')
Path('/tmp/fewclick_readable_ppt_result.json').write_text(json.dumps({'root':str(out),'pptx':str(ppt),'stats':stats}));print(out,stats,flush=True)
