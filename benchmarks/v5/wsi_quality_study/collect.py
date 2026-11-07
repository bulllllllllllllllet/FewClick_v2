"""Independent disk fusion replay and label scoring, outside inference workers."""
import argparse,json,html
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.fair_comparison.metrics import score
from .prepare import dump,labels_for

def mean(values):return float(np.mean(values)) if values else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;c=json.loads((r/'contract.json').read_text());assert sha(r/'plan.json')==c['plan_sha256'];assert sha(r/'coordinate_acceptance.json')==c['coordinate_sha256']
    plans=json.loads((r/'plan.json').read_text())['plans'];audit=json.loads((r/'coordinate_acceptance.json').read_text())['rows'];allrows=[];reports={};gallery=r/'gallery';gallery.mkdir(exist_ok=False)
    for method in ['full','no_match','sam']:
        result=[];timing=[]
        for shard in [0,1]:
            out=r/f'{method}_{shard}';receipt=json.loads((out/'completion.json').read_text());assert receipt['rows_sha256']==sha(out/'rows.json');assert receipt['plan_sha256']==c['plan_sha256'];rows=json.loads((out/'rows.json').read_text());expected=[v['index'] for v in plans if v['index']%2==shard];assert [v['index'] for v in rows]==expected
            for row in rows:
                i=row['index'];item=plans[i];assert row['checkpoint_sha256']==c['checkpoints'][method]['sha256'];assert row['output_sha256']==sha(out/f'roi_{i:03}.npz');assert sha(r/f'labels_{i:03}.png')==audit[i]['label_sha256']
                gt=np.asarray(Image.open(r/f'labels_{i:03}.png'));assert np.array_equal(gt,labels_for(item,item['extent'],item['size']))
                with np.load(out/f'roi_{i:03}.npz') as z:
                    acc=np.zeros((item['size'],item['size']),np.float64);weight=np.zeros_like(acc)
                    assert len(z['tile_probabilities'])==len(item['tiles'])
                    for t,prob in zip(item['tiles'],z['tile_probabilities']):
                        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all();x=(t['dx']+item['radius'])*256;y=(t['dy']+item['radius'])*256;acc[y:y+512,x:x+512]+=prob;weight[y:y+512,x:x+512]+=1
                    assert (weight>0).all() and np.array_equal(weight,z['weight']);fused=(acc/weight).astype(np.float32);assert np.array_equal(fused,z['probability'])
                pred=fused>=.5;valid=gt!=255;truth=(gt==item['task']['target'])&valid;metrics=score(pred,truth,valid,item['mpp'])
                # Independent scalar Dice replay, separate absent-target denominator.
                intersection=np.count_nonzero(pred&truth);assert metrics['tp']==intersection
                if truth.any():assert metrics['dice']==2*intersection/(np.count_nonzero(pred&valid)+np.count_nonzero(truth))
                components,num=label(truth);sizes=np.bincount(components.ravel(),minlength=num+1)[1:];hits=np.bincount(components[pred&truth],minlength=num+1)[1:];detected=int(np.count_nonzero(hits>=.5*sizes));edge_ids=np.unique(np.concatenate([components[0],components[-1],components[:,0],components[:,-1]]));interior=np.array([v not in edge_ids for v in range(1,num+1)])
                distance=None
                if item['support_slide']==item['slide']:
                    a=item['support_context'];x,y,w,h=item['extent'];mx,my=item['geometry']['native_mpp'];b=[x*mx,y*my,(x+w)*mx,(y+h)*my];distance=float(np.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0)))
                rec=dict(method=method,index=i,slide=item['slide'],target=item['task']['target'],metrics=metrics,components=num,detected_components=detected,interior_components=int(interior.sum()),detected_interior_components=int(((hits>=.5*sizes)&interior).sum()),support_query_extent_distance_um=distance)
                result.append(rec);timing.append(row);allrows.append(rec)
                rgb=np.asarray(Image.open(out/f'roi_{i:03}_rgb.png')).copy();overlay=rgb.copy();overlay[pred]=(.55*overlay[pred]+.45*np.array([255,0,0])).astype(np.uint8);Image.fromarray(overlay).save(gallery/f'{method}_{i:03}.png')
                if method=='full':
                    Image.fromarray(rgb).save(gallery/f'rgb_{i:03}.png');gtview=rgb.copy();gtview[truth]=(.55*gtview[truth]+.45*np.array([255,220,0])).astype(np.uint8);gtview[~valid]=128;Image.fromarray(gtview).save(gallery/f'gt_{i:03}.png');Image.open(out/f'roi_{i:03}_support.png').save(gallery/f'support_{i:03}.png')
        positive=[x for x in result if x['metrics']['positive']];absent=[x for x in result if not x['metrics']['positive']];classes=defaultdict(list)
        for x in positive:classes[x['target']].append(x['metrics']['dice'])
        reports[method]=dict(rois=len(result),present=len(positive),absent=len(absent),present_mean_dice=mean([x['metrics']['dice'] for x in positive]),present_class_macro_dice=mean([mean(v) for v in classes.values()]),per_class_dice={k:mean(v) for k,v in classes.items()},bf1=mean([x['metrics']['bf1_um2'] for x in positive if x['metrics']['bf1_um2'] is not None]),absent_FPA=mean([x['metrics']['false_positive_fraction'] for x in absent]),absent_FPA_max=max([x['metrics']['false_positive_fraction'] for x in absent],default=None),components=sum(x['components'] for x in result),detected_components=sum(x['detected_components'] for x in result),timing_rows=timing)
    for report in reports.values():
        rows=report['timing_rows'];tiles=[t for row in rows for t in row['tile_times']]
        report['engineering_costs']=dict(total_query_tiles=len(tiles),support_encoding_sum_seconds=sum(x['support_encode_seconds'] for x in rows),query_IO_sum_seconds=sum(x['io_seconds'] for x in tiles),query_encoding_sum_seconds=sum(x['encode_seconds'] for x in tiles),cached_decoding_sum_seconds=sum(x['cached_decode_seconds'] for x in tiles),cached_decode_median_ms=1000*float(np.median([x['cached_decode_seconds'] for x in tiles])),support_prefix_update_median_ms=1000*float(np.median([x['cached_support_prefix_update_seconds'] for x in rows])),peak_GPU_MiB=max(x['peak_GPU_MiB'] for x in rows),scope='per-window cached decode, not full-WSI click latency; shared devices and OS cache')
    dump(r/'scored_rows.json',allrows);dump(r/'report.json',dict(status='complete_accepted_continuous_WSI_ROI_comparison',slides=len(plans),methods=reports,query_human_clicks=0,plan_sha256=c['plan_sha256'],limitations=['within-cohort spatial ROI validation, not full-slide coverage or blind independent patients','shared GPU/OS cache timing, not cold WSI benchmark','component recall uses >=50% area coverage; cropped components separately recorded','source annotation expert provenance not newly verified']))
    page='<meta charset="utf-8"><title>WSI连续区域公平对比</title><h1>WSI连续区域：支持3正2负，查询零人工点</h1><p>这是空间隔离的连续ROI，不是全片覆盖。黄色GT，红色预测。所有病例保留，缺失目标只统计误报。</p><pre>'+html.escape(json.dumps({k:{a:b for a,b in v.items() if a!='timing_rows'} for k,v in reports.items()},ensure_ascii=False,indent=2))+'</pre>'
    for item in plans:
        i=item['index'];page+=f'<h2>{html.escape(item["slide"])} 目标{item["task"]["target"]}</h2><p>支持区（绿色正点、紫色负点）</p><img width="256" src="gallery/support_{i:03}.png"><div style="display:flex">'
        for key,title in [('rgb','原图'),('gt','GT'),('full','FewClick完整'),('no_match','去匹配'),('sam','SAM适配')]:page+=f'<div style="width:20%">{title}<a href="gallery/{key}_{i:03}.png"><img width="100%" src="gallery/{key}_{i:03}.png"></a></div>'
        page+='</div>'
    (r/'index.html').write_text(page);dump(r/'acceptance.json',dict(status='complete_accepted',report_sha256=sha(r/'report.json'),rows_sha256=sha(r/'scored_rows.json'),roi_method_count=len(allrows)))
if __name__=='__main__':main()
