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
fig,ax=ctx['fig'],ctx['ax'];fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.canvas.get_width_height();dpi=fig.dpi;scale=914400/dpi
out=Path('论文/CVPR_HG_mix/figures')/('editable_ppt_'+datetime.datetime.now().strftime('%m%d_%H%M%S'));out.mkdir()
prs=Presentation();prs.slide_width=Inches(W/dpi);prs.slide_height=Inches(H/dpi);slide=prs.slides.add_slide(prs.slide_layouts[6]);shape_map={};stats={'text':0,'pictures':0,'vector_shapes':0,'groups':0}
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
 else:sh.line.color.rgb=RGBColor(*(round(c*255) for c in edge[:3]));sh.line.width=Pt(width)
def vector(artist,path,face,edge,width,closed):
 polys=path.to_polygons(closed_only=False)
 if not polys:return []
 builder=None
 for poly in polys:
  if len(poly)<2:continue
  xy=[(float(p[0]),float(H-p[1])) for p in poly]
  if builder is None:builder=slide.shapes.build_freeform(xy[0][0],xy[0][1],scale=scale)
  else:builder.move_to(*xy[0])
  builder.add_line_segments(xy[1:],close=closed)
 if builder is None:return []
 sh=builder.convert_to_shape();sh.name=artist.get_gid() or 'vector';style(sh,face,edge,width);stats['vector_shapes']+=1;return [sh]
slide.shapes.turbo_add_enabled=True
for artist in sorted([x for x in ax.get_children() if x.get_gid()],key=lambda x:x.get_zorder()):
 shapes=[]
 if isinstance(artist,Text):
  bb=artist.get_window_extent(renderer);pad=3;sh=slide.shapes.add_textbox(emu(bb.x0-pad),emu(H-bb.y1-2),emu(bb.width+2*pad+4),emu(bb.height+7));sh.name=artist.get_gid();tf=sh.text_frame;tf.clear();tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.TOP
  for i,line in enumerate(artist.get_text().split('\n')):
   p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.alignment=PP_ALIGN.CENTER if artist.get_ha()=='center' else PP_ALIGN.RIGHT if artist.get_ha()=='right' else PP_ALIGN.LEFT;p.space_after=Pt(0);p.space_before=Pt(0);p.line_spacing=1.05
   for r in p.runs:r.font.name='Arial';r.font.size=Pt(artist.get_fontsize());r.font.bold=artist.get_fontweight() in ['bold',700];r.font.color.rgb=RGBColor(*(round(c*255) for c in __import__('matplotlib').colors.to_rgba(artist.get_color())[:3]))
  shapes=[sh];stats['text']+=1
 elif isinstance(artist,AxesImage):
  bb=artist.get_window_extent(renderer);arr=np.asarray(artist.get_array());buf=io.BytesIO();Image.fromarray(arr.astype('uint8')).save(buf,format='PNG');buf.seek(0);sh=slide.shapes.add_picture(buf,emu(bb.x0),emu(H-bb.y1),emu(bb.width),emu(bb.height));sh.name=artist.get_gid();shapes=[sh];stats['pictures']+=1
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
ppt=out/'FewClick_方法主图_单页可编辑.pptx';prs.save(ppt)
# Round-trip parse verifies shape tree and exactly one slide.
check=Presentation(ppt);assert len(check.slides)==1
(out/'validation.json').write_text(json.dumps(dict(stats,slides=1,source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),pptx_sha256=hashlib.sha256(ppt.read_bytes()).hexdigest()),indent=2))
(out/'使用说明.md').write_text('# 单页可编辑方法图\n\n打开PPTX即可编辑；不是整页图片。文字为原生文本框，箭头/区域/特征立方体/token/热图为原生矢量形状，病理图为独立图片。完整组件已组合，可整体移动；需要改细节时进入组或取消组合。正文图1未替换。\n\n沿用已批准的蓝绿橙三栏及校正后的连接；字体使用Arial便于跨机器编辑。页面为宽幅科研图尺寸，可整体缩放后用于论文或汇报。\n')
Path('/tmp/fewclick_ppt_result.json').write_text(json.dumps({'root':str(out),'pptx':str(ppt),'stats':stats}));print(out,stats,flush=True)
