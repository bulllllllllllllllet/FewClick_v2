"""Replace only four outer frame bands, retaining all other raster pixels."""
from pathlib import Path
from PIL import Image,ImageDraw
import io,base64,json,hashlib
src=Path('论文/CVPR_HG_mix/figures/method_final_generated_1115/FewClick_method_final_border_refined.png')
out=src.parent/'vector_frames';out.mkdir(exist_ok=True)
im=Image.open(src).convert('RGB');w,h=im.size
frames=[(7,9,320,800,'#08afe4'),(328,9,1204,538,'#0aaa15'),(329,546,1204,801,'#ff9200'),(1213,9,1940,801,'#ff650b')]
# The replacement is restricted to the narrow perimeter bands, never the panels' interiors.
mask=Image.new('L',im.size,0);md=ImageDraw.Draw(mask)
for x0,y0,x1,y1,c in frames:
 md.rounded_rectangle((x0-4,y0-4,x1+4,y1+4),radius=15,fill=255)
 md.rounded_rectangle((x0+5,y0+5,x1-5,y1-5),radius=5,fill=0)
# Preserve cross-panel connector segments and the named Query R port.
for b in [(306,280,344,298),(307,363,344,378),(307,446,345,460),(1188,331,1233,375),(1188,670,1233,685),(1188,702,1233,720)]:md.rectangle(b,fill=0)
base=Image.composite(Image.new('RGB',im.size,'white'),im,mask)
buf=io.BytesIO();base.save(buf,format='PNG');data=base64.b64encode(buf.getvalue()).decode()
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<image width="{w}" height="{h}" xlink:href="data:image/png;base64,{data}"/>','<g id="editable-outer-frames" fill="none" stroke-width="2.5" stroke-linejoin="round">']
for i,(x0,y0,x1,y1,c) in enumerate(frames):svg.append(f'<rect id="frame-{i+1}" x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="10" stroke="{c}"/>')
svg+=['</g>']
# Keep the existing named port above the outer frame strokes.
ports=[(1189,333,1256,363)]
for i,b in enumerate(ports):
 tile=im.crop(b);bb=io.BytesIO();tile.save(bb,format='PNG');encoded=base64.b64encode(bb.getvalue()).decode()
 svg.append(f'<image id="preserved-port-{i}" x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" xlink:href="data:image/png;base64,{encoded}"/>')
svg+=['</svg>'];(out/'FewClick_method_vector_frames.svg').write_text('\n'.join(svg))
# Antialiased preview: only frame pixels are composited; the remaining image stays byte-identical.
s=4;layer=Image.new('RGBA',(w*s,h*s),(0,0,0,0));d=ImageDraw.Draw(layer)
for x0,y0,x1,y1,c in frames:
 d.rounded_rectangle(((x0-1.25)*s,(y0-1.25)*s,(x1+1.25)*s,(y1+1.25)*s),radius=11.25*s,outline=c,width=10)
layer=layer.resize((w,h),Image.Resampling.LANCZOS)
preview=Image.alpha_composite(base.convert('RGBA'),layer).convert('RGB');
for b in ports:preview.paste(im.crop(b),b[:2])
preview.save(out/'FewClick_method_vector_frames.png')
import numpy as np
changed=np.any(np.asarray(preview)!=np.asarray(im),axis=2);allowed=(np.asarray(mask)>0)|(np.asarray(layer.getchannel('A'))>0)
assert not np.any(changed & ~allowed)
(out/'validation.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'size_px':[w,h],'editable_vector_frames':4,'stroke_width_px':2.5,'corner_radius_px':10,'pixels_changed_outside_frame_bands':int(np.count_nonzero(changed & ~allowed))},indent=2))
(out/'说明.md').write_text('四个外框为独立可编辑SVG矩形，线宽统一2.5px、圆角10px。图内内容沿用原位图，仅清理四个外框附近的窄边带；框外箭头连接处保留。PNG为预览，SVG为矢量边框正式版本。自动校验：边框替换带之外像素差异为0。\n')
print(out)
