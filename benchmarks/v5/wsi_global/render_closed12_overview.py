"""First-manifest WSI qualitative overview from accepted native/GT outputs.

Labels use the original GT-cell-centre/native transform. Display sampling is
never used to recompute metrics. This figure describes a privileged closed12
reference, not the still-pending FewClick/SAM interactive comparison.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import pyvips

from benchmarks.v5.multidomain_data.prepare import LABELS


COLORS = ['#D83A3A','#2E86AB','#FFFFFF','#555B6E','#3BB273','#9BD1A2',
          '#8E6C8A','#F29559','#5B3F8B','#69C9C2','#F2C14E','#A61B5B']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run(a):
    started=time.perf_counter(); root=a.root
    data=root/'main30_full_1027_171918'; source=root/'closed12_raster_full_1027_221200'
    accepted=root/'closed12_raster_acceptance_1027_221200'; summary=read(source/'summary.json'); receipt=read(accepted/'summary.json')
    require(receipt['status']=='accepted_closed12_full30_native_raster_and_GT_scores' and receipt['source_summary_sha256']==sha(source/'summary.json'), 'independently accepted all30 native output')
    require(read(data/'summary.json')['slides_sha256']==sha(data/'slides_private.json') and summary['signature']['data_summary_sha256']==sha(data/'summary.json'), 'same fixed original30 provenance')
    require(sha(source/'rows.json')==summary['rows_sha256'] and sha(source/'items.json')==summary['items_sha256'], 'complete unchanged accepted rows/items')
    record=read(data/'slides_private.json')[0]; sid=record['wsi_id']
    item=next(r for r in read(source/'items.json') if r['wsi_id']==sid)
    rows=[r for r in read(source/'rows.json') if r['wsi_id']==sid]
    require({r['target'] for r in rows}==set(range(12)), 'first fixed WSI with all accepted class rows')
    q=next(q for q in read(root/'support_query_full_1027_173800/query_inputs.json') if q['wsi_id']==sid)
    labels_path=data/sid/'labels.npy'; prediction=item['prediction_files']['class_prediction.npy']
    require(sha(labels_path)==record['files'][str(labels_path)]['sha256'] and sha(prediction['path'])==prediction['sha256'], 'same actual originalGT and accepted native categorical bytes')
    w,h=record['native_wh']; gw,gh=record['gt_wh']; mx,my=record['native_mpp']
    require(a.height>=256 and a.height<=2048, 'bounded diagnostic display size')
    dh=a.height; dw=round(dh*w/h)
    gx=np.floor((np.arange(dw)+.5)*gw/dw).astype(np.int64)
    gy=np.floor((np.arange(dh)+.5)*gh/dh).astype(np.int64)
    x0,y0,x1,y1=item['GT_domain']['GT_cell_bounds'];left,top,right,bottom=item['coverage']['bounds_native']
    valid=(gy[:,None]>=y0)&(gy[:,None]<y1)&(gx[None,:]>=x0)&(gx[None,:]<x1)
    labels=np.load(labels_path,mmap_mode='r'); raw_gt=np.asarray(labels[np.ix_(gy,gx)])
    valid &= raw_gt!=255
    nx=np.floor((gx+.5)*w/gw).astype(np.int64)-left
    ny=np.floor((gy+.5)*h/gh).astype(np.int64)-top
    pred=np.load(prediction['path'],mmap_mode='r')
    # Out-of-domain locations are display-only unknowns; never negatives.
    display_gt=np.full((dh,dw),255,np.uint8); display_pred=np.full_like(display_gt,255)
    yy,xx=np.nonzero(valid);display_gt[yy,xx]=raw_gt[yy,xx]
    display_pred[yy,xx]=pred[ny[yy],nx[xx]]
    pyvips.cache_set_max(0)
    image=pyvips.Image.new_from_file(record['source_wsi'],access='sequential')
    require([image.width,image.height]==[w,h] and image.bands>=3, 'actual original H&E native header')
    small=image.extract_band(0,n=3).resize(dw/w,vscale=dh/h,kernel='linear').cast('uchar')
    require((small.width,small.height,small.bands)==(dw,dh,3), 'actual bounded H&E thumbnail geometry')
    rgb=np.frombuffer(small.write_to_memory(),np.uint8).reshape(dh,dw,3).copy()
    a.output.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(a.output/'HE_thumbnail.png')
    for name,value in [('GT_display.npy',display_gt),('prediction_display.npy',display_pred),('valid_display.npy',valid),('GT_x_indices.npy',gx),('GT_y_indices.npy',gy)]:
        np.save(a.output/name,value,allow_pickle=False)
    palette=np.array([tuple(int(c[i:i+2],16) for i in (1,3,5)) for c in COLORS],np.uint8)
    def colorize(values):
        out=np.full((*values.shape,3),228,np.uint8);take=values!=255;out[take]=palette[values[take]];return out
    extent=[0,w*mx/1000,h*my/1000,0]
    macro=math.fsum(r['dice'] for r in rows if r['target']!=2 and r['dice'] is not None)/sum(r['target']!=2 and r['dice'] is not None for r in rows)
    fat=next(r for r in rows if r['target']==10)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(11.8,5.7),sharex=True,sharey=True)
    panels=[rgb,colorize(display_gt),colorize(display_pred)]
    titles=['A  H&E and fixed evaluation domain','B  Original GT (evaluated domain)','C  Closed12 CE reference (no clicks)']
    for ax,array,title in zip(axes,panels,titles):
        ax.imshow(array,extent=extent,interpolation='nearest',aspect='equal');ax.set_title(title,loc='left',fontsize=10,pad=9)
        ax.set_xlabel('x (mm)');ax.tick_params(labelsize=8);ax.set_xlim(0,extent[1]);ax.set_ylim(extent[2],0)
        ax.add_patch(Rectangle((left*mx/1000,top*my/1000),(right-left)*mx/1000,(bottom-top)*my/1000,fill=False,edgecolor='#161616',linewidth=.8))
    axes[0].set_ylabel('y (mm)')
    axes[0].axvspan(0,q['source_cut_native']*mx/1000,color='#177E89',alpha=.13)
    axes[0].axvline(q['source_cut_native']*mx/1000,color='#177E89',linewidth=1,linestyle='--')
    axes[0].text(.15,.25,'Shaded left band: reserved support region',fontsize=7,va='top',color='#13565D',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    handles=[Patch(facecolor=COLORS[c],edgecolor='#666666',linewidth=.4,label=str(c)+' '+LABELS['crc'][c].replace('_',' ')) for c in range(12)]
    handles.append(Patch(facecolor='#E4E4E4',edgecolor='#666666',linewidth=.4,label='Outside scored domain / void'))
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=7,frameon=False,bbox_to_anchor=(.5,.025),columnspacing=1.5)
    fig.suptitle('Fixed first WSI: '+sid,fontsize=11,y=.995)
    fig.text(.5,.952,f'Closed12 reference: slide tissue macro Dice {macro:.4f}  |  Fat Dice {fat["dice"]:.4f}  |  Full-domain accepted scores',ha='center',fontsize=8)
    fig.subplots_adjust(left=.055,right=.99,top=.91,bottom=.23,wspace=.16)
    fig.savefig(a.output/'closed12_overview.png',dpi=240,bbox_inches='tight')
    fig.savefig(a.output/'closed12_overview.pdf',dpi=300,bbox_inches='tight');plt.close(fig)
    metadata=dict(wsi_id=sid,selection='first original slides_private.json record, before looking at scores',native_wh=[w,h],GT_wh=[gw,gh],native_mpp=[mx,my],display_wh=[dw,dh],GT_bounds=[x0,y0,x1,y1],native_bounds=[left,top,right,bottom],extent_mm=extent,source_cut_native=q['source_cut_native'],RGB_source=record['source_wsi'],RGB_file_stat=dict(bytes=Path(record['source_wsi']).stat().st_size,mtime_ns=Path(record['source_wsi']).stat().st_mtime_ns),GT_path=str(labels_path),GT_sha256=sha(labels_path),prediction_file=prediction,palette=COLORS,accepted_rows=rows,slide_tissue_macro_dice=macro,fat_dice=fat['dice'],metric_role='All numbers inherited from accepted full native/GT confusion rows; none recomputed on display sampling',RGB_role='Linear display resampling of original H&E; original RGB file not rehashed in this rendering step',alignment_scope='Original accepted GT extent affine and GT-cell-centre/native projection, not independently proven anatomical registration')
    (a.output/'mapping.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    caption=f'''# 首张固定WSI的闭集参考概览\n\n清单第1张 `{sid}`，由清单顺序选取。A为原H&E及评价范围：左侧蓝色带为交互协议预留的支持区域，黑框为已接受的严格远距输出矩形。B/C使用同一坐标范围、同一GT像元中心到原生像素映射和同一12标签配色；灰色为评分范围以外/void，背景ID2为白色。闭集模型使用12类CE监督和组织ID、无需点击。\n\n图中数字为该单张切片的已接受完整远距域评分：11组织macro Dice {macro:.6f}，fat Dice {fat['dice']:.6f}；不是从缩略图计算。30张整体WSI等权Dice另见已接受描述性报告。缩略显示使用{dw}×{dh}采样，细小区域可能不可见；原GT extent affine保留，独立解剖配准未证明。\n\n这是闭集参考的定性输出，FewClick/SAM交互完整结果仍待接受；该单张概览不提供模型优势或总体泛化结论。映射、原始评分行、数据/接受SHA和固定选择规则见mapping.json/summary.json。图像可供人工检查，未宣称用户已完成视觉接受。\n'''
    (a.output/'caption_zh.md').write_text(caption)
    result=dict(status='rendered_closed12_first_WSI_overview_pending_mapping_audit',worker_pid=os.getpid(),source_sha256=sha(__file__),native_acceptance_sha256=sha(accepted/'summary.json'),native_source_summary_sha256=sha(source/'summary.json'),data_summary_sha256=sha(data/'summary.json'),wsi_id=sid,files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},seconds=time.perf_counter()-started,scope='Qualitative first-manifest slide from accepted privileged closed12 native output; no new model inference, sampled metric or paper modification.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--height',type=int,default=1024);run(p.parse_args())
