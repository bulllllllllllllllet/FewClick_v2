"""Extract complete drawing calls from original vector source, preserving editability."""
from pathlib import Path
import datetime,hashlib,json,copy,xml.etree.ElementTree as ET,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.transforms import Bbox
from PIL import Image,ImageDraw,ImageFont
src=Path('benchmarks/v5/paper_figures/method_vector.py');code=src.read_text();split=code.index("panel(6,7,");end=code.index('# export, validate')
ctx={'__name__':'element_extraction','__file__':str(src)};exec(compile(code[:split],str(src),'exec'),ctx)
records=[];depth=[0]
functions=['panel','cube','tokens','regions','enc','heat','im','box','plus','click','line']
def wrap(name,fn):
 def call(*args,**kwargs):
  top=depth[0]==0;before=set(ctx['ax'].get_children()) if top else None;depth[0]+=1
  try:return fn(*args,**kwargs)
  finally:
   depth[0]-=1
   if top:records.append(dict(kind=name,args=args,kwargs=kwargs,artists=[a for a in ctx['ax'].get_children() if a not in before]))
 return call
for name in functions:ctx[name]=wrap(name,ctx[name])
exec(compile(code[split:end],str(src),'exec'),ctx)
fig,ax=ctx['fig'],ctx['ax'];fig.canvas.draw();renderer=fig.canvas.get_renderer();all_artists=[a for a in ax.get_children() if a.get_gid()];original={a:a.get_visible() for a in all_artists}
# Add intact multi-object structures that were drawn directly in the source.
for title,bounds in [('cross_scale_tree',(1000,245,1165,492)),('four_tiles',(455,100,535,179)),('whole_tiled_fusion',(320,62,1168,211)),('binary_mask',(1850,511,1921,584)),('positive_token',(1640,110,1712,124)),('negative_token',(1640,157,1712,171))]:
 x0,y0,x1,y1=bounds;p0=ax.transData.transform((x0,y1));p1=ax.transData.transform((x1,y0));area=Bbox.from_extents(*p0,*p1);arts=[]
 for a in all_artists:
  b=a.get_window_extent(renderer)
  if np.isfinite(b.extents).all() and b.width>0 and b.height>0 and b.x0>=area.x0-1 and b.x1<=area.x1+1 and b.y0>=area.y0-1 and b.y1<=area.y1+1:arts.append(a)
 assert arts,title
 records.append(dict(kind='group',args=(0,0,title),kwargs={},artists=arts))
# Save complete groups only; avoid tiny line fragments inside cubes/token banks.
root=Path('论文/CVPR_HG_mix/figures')/('elements_'+datetime.datetime.now().strftime('%m%d_%H%M%S'));root.mkdir();(root/'SVG').mkdir();(root/'PNG').mkdir();(root/'原始病理图片').mkdir()
labels={'group':'组合模块','cube':'特征立方体','tokens':'区域token','regions':'软区域划分','enc':'编码器','heat':'热图矩阵','im':'病理图像','box':'功能模块','plus':'运算符','click':'正负点击','line':'连接箭头'}
seen=set();selected=[];counts={};manifest=[]
def serial(v):
 if isinstance(v,np.ndarray):return {'array_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'shape':list(v.shape)}
 if isinstance(v,(list,tuple)):return [serial(a) for a in v]
 if isinstance(v,dict):return {k:serial(a) for k,a in v.items()}
 return v
for rec in records:
 kind=rec['kind'];args=rec['args'];kw=rec['kwargs']
 if kind=='panel':continue
 if kind=='line':
  # Only keep arrows, including complete orthogonal multi-segment connectors.
  if kw.get('arrow',True)==False:continue
  points=np.array(args[0]);norm=(points-points[0]).tolist();sig=[norm,serial(args[1:]),kw]
 else:
  sig=[serial(args[1:]) if kind=='im' else serial(args[2:]),kw]
  if kind=='im':sig=[serial(args[0]),serial(args[3:]),kw]
 key=json.dumps([kind,sig],sort_keys=True,ensure_ascii=False)
 if key in seen:continue
 seen.add(key);counts[kind]=counts.get(kind,0)+1;rec['name']=f'{kind}_{counts[kind]:02d}';selected.append(rec)
for a in all_artists:a.set_visible(False)
SVGNS='http://www.w3.org/2000/svg';ET.register_namespace('',SVGNS);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
for i,rec in enumerate(selected):
 artists=rec['artists'];name=rec['name'];kind=rec['kind']
 for a in artists:a.set_visible(True)
 bboxes=[]
 for a in artists:
  b=a.get_window_extent(renderer)
  if np.isfinite(b.extents).all() and b.width>=0 and b.height>=0:bboxes.append(b)
 if not bboxes:raise RuntimeError(name)
 b=Bbox.union(bboxes);b=Bbox.from_extents(b.x0-7,b.y0-7,b.x1+7,b.y1+7).transformed(fig.dpi_scale_trans.inverted())
 svg=root/'SVG'/f'{name}.svg';png=root/'PNG'/f'{name}.png'
 fig.savefig(svg,bbox_inches=b,transparent=True,pad_inches=0);fig.savefig(png,bbox_inches=b,transparent=True,pad_inches=0,dpi=240)
 # Group every selected sub-element under ONE top-level group for easy copy/paste.
 tree=ET.parse(svg);r=tree.getroot();ns={'s':SVGNS};axis=r.find('.//s:g[@id="axes_1"]',ns)
 if axis is not None:
  g=ET.Element('{'+SVGNS+'}g',{'id':name,'data-component':labels[kind]})
  for ch in list(axis):axis.remove(ch);g.append(ch)
  axis.append(g);tree.write(svg,encoding='utf-8',xml_declaration=True)
 img=Image.open(png).convert('RGBA');assert img.getchannel('A').getextrema()[0]==0,name
 manifest.append({'id':name,'category':labels[kind],'svg':str(svg.relative_to(root)),'png':str(png.relative_to(root)),'vector_text_count':len(r.findall('.//s:text',ns)),'raster_images':len(r.findall('.//s:image',ns)),'objects':len(artists),'pixels':list(img.size),'source_ids':[a.get_gid() for a in artists]})
 if kind=='im':Image.fromarray(rec['args'][0]).save(root/'原始病理图片'/f'{name}.png')
 for a in artists:a.set_visible(False)
 if (i+1)%15==0:print('exported',i+1,'/',len(selected),flush=True)
# Contact sheets are previews, not edits to any source image.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for page in range((len(manifest)+29)//30):
 items=manifest[page*30:(page+1)*30];sheet=Image.new('RGB',(1500,((len(items)+4)//5)*205),'#eef1f4');draw=ImageDraw.Draw(sheet)
 for k,item in enumerate(items):
  x=(k%5)*300;y=(k//5)*205;draw.rounded_rectangle((x+7,y+7,x+293,y+198),radius=8,fill='white');im=Image.open(root/item['png']).convert('RGBA');im.thumbnail((270,145));sheet.paste(im,(x+(300-im.width)//2,y+18+(145-im.height)//2),im);draw.text((x+14,y+172),item['id'],font=font,fill='#203344')
 sheet.save(root/f'预览索引_{page+1:02d}.jpg')
(root/'manifest.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'counts':counts,'assets':manifest},ensure_ascii=False,indent=2))
readme='''# FewClick方法图独立素材包

每个组件已拆成单独文件，并在SVG内组合成一个完整组。

- `SVG/`：可编辑矢量素材。打开或导入Inkscape、Illustrator后，可整体复制、移动、缩放；需要改内部细节时进入组或取消组合。
- `PNG/`：同名透明背景图片，可直接放入PPT或绘图工具。
- `原始病理图片/`：不带框线的原始图像插入素材。
- `预览索引_*.jpg`：按文件名查找素材。
- `manifest.json`：各素材的来源对象、类型及文件路径。

文件名前缀：cube=特征立方体，tokens=区域token，regions=软区域划分，enc=编码器，heat=概率/对应/残差示意图，im=病理图片，box=功能模块，plus=运算符，click=正负点击，line=完整连接箭头。

这些素材取自已有SVG的绘图对象，不是从整张预览图截图。文字保持可编辑，病理图片保持内嵌位图；热图和区域示意可编辑但不是实测激活。相同组件去重保留，未修改原主图。
''';(root/'使用说明.md').write_text(readme)
archive=root.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(root.rglob('*')):
  if p.is_file():z.write(p,Path(root.name)/p.relative_to(root))
print('OUTPUT',root,'ASSETS',len(manifest),'COUNTS',counts,'ZIP',archive,flush=True)
Path('/tmp/fewclick_elements_result.json').write_text(json.dumps({'root':str(root),'zip':str(archive),'count':len(manifest),'counts':counts}))
