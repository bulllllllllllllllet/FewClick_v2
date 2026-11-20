"""Normalize the full editable diagram to a smaller authoring canvas."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import zipfile,json,hashlib
from lxml import etree as ET
src=Path('论文/CVPR_HG_mix/figures/readable_ppt_1115_013308/FewClick_AAAI比例大字版_可编辑.pptx')
out=src.parent.parent/('arial9pt_ppt_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m%d_%H%M%S'));out.mkdir()
ppt=out/'FewClick_Arial9-10pt_可编辑.pptx'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
factor=16/24
with zipfile.ZipFile(src) as z:contents={n:z.read(n) for n in z.namelist()}
for name,data in list(contents.items()):
 if not name.endswith('.xml'):continue
 root=ET.fromstring(data)
 if name=='ppt/presentation.xml':
  for el in root.xpath('//p:sldSz',namespaces=ns):
   for k in ('cx','cy'):el.set(k,str(round(int(el.get(k))*factor)))
 if name.startswith('ppt/slides/slide'):
  for xfrm in root.xpath('//a:xfrm',namespaces=ns):
   for el in xfrm:
    for k in ('x','y','cx','cy'):
     if el.get(k) is not None:el.set(k,str(round(int(el.get(k))*factor)))
  for el in root.xpath('//a:rPr|//a:defRPr|//a:endParaRPr',namespaces=ns):
   size=1000 if el.get('b')=='1' or int(el.get('sz','900'))>=2000 else 900
   el.set('sz',str(size))
   for tag in ('latin','ea','cs'):
    face=el.find('a:'+tag,ns)
    if face is None:face=ET.SubElement(el,'{'+ns['a']+'}'+tag)
    face.set('typeface','Arial')
  for line in root.xpath('//a:ln',namespaces=ns):
   if line.find('a:noFill',ns) is None:
    old=int(line.get('w','9525'));line.set('w',str(12700 if old>=12700 else 9525))
 contents[name]=ET.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
with zipfile.ZipFile(ppt,'w',zipfile.ZIP_DEFLATED) as z:
 for n,data in contents.items():z.writestr(n,data)
(out/'使用说明.md').write_text('画布：16 × 6.798英寸（406.4 × 172.68毫米），保留全部结构与组件分组。\n字体：Arial，正文9pt、标题和强调标记10pt。可见线条：0.75–1pt。\n这些为PPT原尺寸下的字号和线宽；插入论文后整体缩放会同比缩小。\n',encoding='utf-8')
Path('/tmp/fewclick_arial_ppt_result.json').write_text(json.dumps({'pptx':str(ppt),'root':str(out)},ensure_ascii=False))
print(ppt)
