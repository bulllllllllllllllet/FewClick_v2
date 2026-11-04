"""Read-only source reconciliation, full current-label audit and review dashboard."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import yaml

from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS
from benchmarks.v5.data_review.gallery import NAMES, COLORS
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch

BASE = Path('/nfs-medical3/zyh/fewclick_v2')
COHORT = Path('/nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet')
OLD = Path('/nfs-medical3/zyh/v4/phase1/data/tiled_patch_index_cohort200_20260716_013248/patch_index_10x_tiled_gt.parquet')
CRC = BASE/'extensible_prompt/tissue_spatial_split_0922_001235/patches_index.parquet'


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intersections(rows, footprint):
    groups = defaultdict(list)
    for r in rows: groups[r['space_key']].append(r)
    pairs = []
    for group in groups.values():
        train = [r for r in group if r['split']=='train']; val = [r for r in group if r['split']=='val']
        for t in train:
            for v in val:
                a, b = t[footprint], v[footprint]
                w, h = min(a[2],b[2])-max(a[0],b[0]), min(a[3],b[3])-max(a[1],b[1])
                if w>1e-6 and h>1e-6:
                    pairs.append(dict(train=t['tile_id'],val=v['tile_id'],space=t['space_key'],intersection_um2=w*h))
    return pairs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(exist_ok=False); assets=a.output/'assets';assets.mkdir();started=time.monotonic()
    pyvips.cache_set_max(0)
    c=pd.read_parquet(COHORT); old=pd.read_parquet(OLD); crc=pd.read_parquet(CRC)
    _, catalog=load_cell_catalog(TASKS); original={r['wsi_id']:r for r in c.to_dict('records')}
    # Unique sources, separate representation inventories: never concatenate duplicate sources.
    current=[]; aggregates=defaultdict(Counter); class_tiles=defaultdict(Counter); duplicates=defaultdict(list)
    for index,r in enumerate(catalog.values()):
        gt=np.asarray(Image.open(r['labels'])); assert gt.shape==(512,512)
        assert hashlib.sha256(gt.tobytes()).hexdigest()==r['labels_array_sha256']
        values,counts=np.unique(gt,return_counts=True); ds=r['dataset'];split=r['split']
        assert set(values).issubset(set(range(len(NAMES[ds])))|{255})
        actual={str(v):int(n) for v,n in zip(values,counts)}
        assert actual==json.loads(r['counts_json']),r['tile_id']
        aggregates[ds+'|'+split].update({int(k):v for k,v in actual.items()});class_tiles[ds+'|'+split].update(map(int,values))
        multi=json.loads(r['multiscale_json']); native=multi['native_boxes'];mpp=multi['native_mpp']
        fine=[r[k] for k in ['x0_um','y0_um','x1_um','y1_um']]
        # The crop-to-global offset derives from the known fine footprint; TIGER boxes are ROI-local.
        b=native[2]; f=native[0]
        context=[fine[0]+(b[0]-f[0])*mpp[0],fine[1]+(b[1]-f[1])*mpp[1],
                 fine[0]+(b[0]+b[2]-f[0])*mpp[0],fine[1]+(b[1]+b[3]-f[1])*mpp[1]]
        assert np.allclose(context,multi['context_um'],rtol=0,atol=1e-6)
        oid=r['space_key'].split('|',1)[1] if ds=='crc' else None
        assert ds!='crc' or oid in original
        record=dict(tile_id=r['tile_id'],dataset=ds,split=split,space_key=r['space_key'],source_id=r['source_id'],
                    historical_split=original[oid]['split'] if oid else 'not_original_CRC_cohort',
                    fine_um=fine,context4x_um=context,spatial_group_id=r['spatial_group_id'],labels_sha256=r['labels_array_sha256'])
        current.append(record);duplicates[(r['space_key'],*fine)].append(r['tile_id'])
        if index%100==0:print(json.dumps(dict(stage='label_audit',done=index+1,total=len(catalog))),flush=True)
    fine_overlaps=intersections(current,'fine_um');context_overlaps=intersections(current,'context4x_um')
    exact_duplicates=[ids for ids in duplicates.values() if len(ids)>1]
    pd.DataFrame(current).to_json(a.output/'current_fields.jsonl',orient='records',lines=True,force_ascii=False)
    dump(a.output/'overlap_audit.json',dict(fine=fine_overlaps,context4x=context_overlaps,exact_duplicates=exact_duplicates,
                                          scope='current1737 fine and actual4x rectangular RGB footprints; no patient-disjoint claim; cross-panel/rescan/extra cell crop footprints not covered'))
    use=defaultdict(Counter)
    for r in current:
        if r['dataset']=='crc':use[r['space_key'].split('|',1)[1]][r['split']]+=1
    old_counts=old.groupby('wsi_id').size().to_dict();crc_counts=crc.groupby('wsi_id').size().to_dict()
    lineage=[dict(wsi_id=r['wsi_id'],historical_split=r['split'],historical_patches=int(old_counts.get(r['wsi_id'],0)),
                  spatial_candidate_patches=int(crc_counts.get(r['wsi_id'],0)),current_train=use.get(r['wsi_id'],Counter())['train'],current_val=use.get(r['wsi_id'],Counter())['val']) for r in c.to_dict('records')]
    assert len(use)==167 and sum(r['current_train']+r['current_val'] for r in lineage)==1500
    pd.DataFrame(lineage).to_csv(a.output/'cohort200_lineage.csv',index=False)
    # Keep the original twenty examples fixed; add ten examples from the historical representation.
    previous=json.loads((BASE/'cvpr_research/dataset_gallery_1104_130305/selection.json').read_text())
    rng=random.Random(20261104); selected_old=[]
    for split in ['train','val']:
        pool=old[old.split==split].sort_values('patch_id').to_dict('records');selected_old.extend(rng.sample(pool,5))
    dump(a.output/'selection.json',dict(seed=20261104,current=previous['tiles'],historical=[dict(patch_id=r['patch_id'],wsi_id=r['wsi_id'],split=r['split']) for r in selected_old],
                                      policy='current samples held fixed; historical five train/five val sampled uniformly by patch before pixel reads; no original test pixel reads; no score filtering'))
    config=yaml.safe_load(Path('benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml').read_text())
    cards=[]; image_audit=[]
    def card(rgb,gt,key,group,split,title,meta,contexts=None):
        assert rgb.shape==(512,512,3) and gt.shape==(512,512)
        colored=np.zeros_like(rgb)
        for label in np.unique(gt):colored[gt==label]=[40,40,40] if label==255 else COLORS[int(label)]
        images=[('原图',Image.fromarray(rgb)),('组织标签',Image.fromarray(colored)),('标签叠加',Image.blend(Image.fromarray(rgb),Image.fromarray(colored),.4))]
        if contexts:images+=contexts
        figures=[]
        for n,(caption,image) in enumerate(images):
            filename=key+'_'+str(n)+'.png';image.save(assets/filename)
            assert np.array_equal(np.asarray(Image.open(assets/filename)),np.asarray(image))
            figures.append(f'<figure><img loading="lazy" src="assets/{filename}" onclick="zoom(this)"><figcaption>{html.escape(caption)}</figcaption></figure>')
        ds='tiger' if group=='tiger' else 'crc';values,counts=np.unique(gt,return_counts=True)
        legend=' '.join(f'<span><i style="background:rgb({",".join(map(str,[40,40,40] if int(v)==255 else COLORS[int(v)]))})"></i>{html.escape("忽略255" if v==255 else str(v)+" "+NAMES[ds][int(v)])} {n/gt.size:.1%}</span>' for v,n in zip(values,counts))
        cards.append(f'<article class="sample" data-group="{group}" data-split="{split}"><h3>{html.escape(title)}</h3><div class="legend">{legend}</div><div class="images">'+''.join(figures)+'</div><details><summary>查看来源和坐标</summary><pre>'+html.escape(json.dumps(meta,ensure_ascii=False,indent=2))+'</pre></details></article>')
        image_audit.append(dict(key=key,group=group,split=split,rgb_array_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),label_array_sha256=hashlib.sha256(gt.tobytes()).hexdigest()))
    for i,r in enumerate(selected_old):
        source=pyvips.Image.new_from_file(r['wsi_path'],access='random').crop(*[int(r[k]) for k in ['x_level0','y_level0','width_level0','height_level0']]).extract_band(0,n=3)
        native=np.frombuffer(source.write_to_memory(),dtype=np.uint8).reshape(source.height,source.width,3)
        rgb=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.LANCZOS))
        gt=decode_gt_patch(Path(r['gt_path']),*[int(r[k]) for k in ['x_10x','y_10x','width_10x','height_10x']],config['data']['class_map'],255)
        native_gt=pyvips.Image.new_from_file(r['gt_path'],access='random').crop(*[int(r[k]) for k in ['x_10x','y_10x','width_10x','height_10x']]).extract_band(0,n=3)
        rgb_gt=np.frombuffer(native_gt.write_to_memory(),dtype=np.uint8).reshape(512,512,3).astype(np.uint32)
        packed=rgb_gt[...,0]*65536+rgb_gt[...,1]*256+rgb_gt[...,2]
        decoded=np.full((512,512),255,np.uint8)
        for item in config['data']['class_map']:
            red,green,blue=item['rgb'];decoded[packed==red*65536+green*256+blue]=item['id']
        assert np.array_equal(gt,decoded), 'Independent RGB palette decode must match original loader'

        card(rgb,gt,'original_'+str(i),'original',r['split'],'原主队列历史 patch · '+r['split']+' · '+r['wsi_id'],dict(patch_id=r['patch_id'],source=r['wsi_path'],label=r['gt_path'],historical_split=r['split'],coordinates_level0=[int(r[k]) for k in ['x_level0','y_level0','width_level0','height_level0']],note='历史patch表现形式，未新增到当前训练；并非第三独立数据集'))
        print(json.dumps(dict(stage='historical_visuals',done=i+1,total=10)),flush=True)
    cache=BASE/'cvpr_research/cell_repair_rebalance_1102_210552/normalized';entries={r['tile_id']:r for r in map(json.loads,(cache/'index.jsonl').read_text().splitlines())}
    current_by_id={r['tile_id']:r for r in current}
    for s in previous['tiles']:
        r=catalog[s['tile_id']];entry=entries[r['tile_id']];assert sha(entry['file'])==entry['sha256']
        with np.load(entry['file']) as f:images=[f['image_'+str(i)].copy() for i in range(3)]
        rgb=np.asarray(Image.open(json.loads(r['variants_json'])['original']['image']).convert('RGB'));assert np.array_equal(rgb,images[0])
        gt=np.asarray(Image.open(r['labels']))
        card(rgb,gt,r['tile_id'],r['dataset'],r['split'],r['dataset'].upper()+' 当前视野 · '+r['split']+' · '+r['tile_id'],current_by_id[r['tile_id']],contexts=[('2×上下文',Image.fromarray(images[1])),('4×上下文',Image.fromarray(images[2]))])
    summaries={k:dict(pixel_counts=dict(v),tile_occurrences=dict(class_tiles[k])) for k,v in aggregates.items()}
    summary=dict(status='complete_readonly_data_review',cohort200=len(c),historical_patch_rows=len(old),CRC_candidate_rows=len(crc),
                 current_counts=dict(Counter(r['dataset']+'|'+r['split'] for r in current)),current_CRC_source_slides=len(use),
                 CRC_candidate_unique_slides=crc.wsi_path.nunique(),CRC_exact_path_overlap=len(set(crc.wsi_path)&set(c.wsi_path)),
                 fine_overlap_pairs=len(fine_overlaps),context4x_overlap_pairs=len(context_overlaps),exact_duplicate_groups=len(exact_duplicates),
                 all1737_label_hash_shape_ID_counts_pass=True,class_summary=summaries,original_test_pixels_read=False,
                 source_hashes={str(p):sha(p) for p in [COHORT,OLD,CRC,Path(__file__)]},training_paused=True,
                 limitations=['仅当前1737输入矩形范围；全53534候选空间重划分未执行','来源ID去重不能排除重扫描或跨模态重复','原CRC标签生成和专家审核来源仍待确认','目视查看历史train/val样本不构成独立盲测','本次不改训练清单、不合并标签、不启动训练'])
    dump(a.output/'summary.json',summary);dump(a.output/'image_audit.json',image_audit)
    def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    class_rows=[]
    for key,v in summaries.items():
        ds=key.split('|')[0];total=sum(v['pixel_counts'].values())
        for label in [*range(len(NAMES[ds])),255]:
            n=v['pixel_counts'].get(label,0);class_rows.append([key,label,'忽略' if label==255 else NAMES[ds][label],v['tile_occurrences'].get(label,0),f'{n/total:.2%}',n])
    lineage_html=table(['WSI','原划分','历史patch数','空间候选数','当前train','当前val'],[[r[k] for k in ['wsi_id','historical_split','historical_patches','spatial_candidate_patches','current_train','current_val']] for r in lineage])
    # Per-slide actual-context geometry, both splits retained and scaled together.
    maps=[]
    for space in sorted({r['space_key'] for r in current}):
        items=[r for r in current if r['space_key']==space];boxes=np.array([r['context4x_um'] for r in items]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0);scale=560/max(hi-lo)
        rects=[]
        for r in items:
            b=r['context4x_um'];color='#176ec1' if r['split']=='train' else '#e77514'
            rects.append(f'<rect x="{(b[0]-lo[0])*scale+10}" y="{(b[1]-lo[1])*scale+10}" width="{(b[2]-b[0])*scale}" height="{(b[3]-b[1])*scale}" fill="{color}" fill-opacity=".12" stroke="{color}" stroke-width=".8"><title>{r["tile_id"]} {r["split"]}</title></rect>')
        maps.append('<details><summary>'+html.escape(space)+f'（{len(items)}视野）</summary><svg viewBox="0 0 580 580" width="580">'+''.join(rects)+'</svg></details>')
    overview=table(['范围','癌种/来源','规模','说明'],[['原主队列','广东省人民医院结直肠癌','200 WSI；原140/30/30','源库存，非第三数据集'],['当前CRC','同一主队列','1500视野；167 WSI','原主队列子集＋新空间划分'],['TIGER','公开乳腺癌ROI','237视野；156 train/81 val','独立域；7类源标签，当前不保证全类覆盖']])
    warning=f'当前1737范围：fine跨train/val相交 {len(fine_overlaps)} 对；实际4×输入相交 {len(context_overlaps)} 对；精确重复范围 {len(exact_duplicates)} 组。仅有空间相交统计，不由同患者或同片直接认定泄漏。'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>FewClick 统一数据检查</title><style>body{font:16px system-ui;background:#f1f5f9;color:#18283d;margin:24px}header,section,article{background:white;padding:20px;border-radius:12px;margin:18px 0}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2}a{color:#1261b0;margin-right:16px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}thead{background:#e7eff8}.images{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0;width:calc(33.3% - 10px);min-width:230px;max-width:512px}img{width:100%;cursor:zoom-in}figcaption{font-size:14px;padding:5px}.legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}i{display:inline-block;width:15px;height:15px;margin-right:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}select,button{padding:8px;font-size:16px}dialog{max-width:95vw}dialog img{max-height:85vh;width:auto;max-width:90vw}.notice{padding:16px;background:#fff2cd;border-left:4px solid #d69712}details{margin:12px 0}svg{max-width:100%;background:#f7f7f7}</style><nav><a href="#overview">来源关系</a><a href="#classes">类别分布</a><a href="#gallery">随机图像</a><a href="#spatial">空间划分</a><a href="#lineage">200张覆盖</a></nav><header><h1>FewClick 统一数据检查</h1><p>训练已暂停。本页把原主队列、当前CRC子集和TIGER按来源关联，供人工核查；数据仍保留原样。</p><div class="notice">纠正：原主队列与当前CRC同源，不能算两个独立数据集再拼接。标签是否准确、组织本体是否合理，请结合原图检查；技术校验通过不等于标注正确。</div></header>'''
    page+='<section id="overview"><h2>来源与使用范围</h2>'+overview+'<p>CRC空间候选来自原train140＋val30，未含原test30。当前train/val是空间划分，不能与原WSI划分混同；保留患者/切片来源，不强制患者互斥。</p><p>当前CRC12类；TIGER源7类单独保存。数字ID不跨域合并。CRC 2为有效背景负例；TIGER原0在当前预处理映射为255忽略，7为其他复合组织；不可把未标注当背景。TIGER当前缺少的类别必须显示0。</p><p>原CRC参考标签的生成/专家审核来源尚待确认。TIGER采用公开ROI标注，当前未下载完整WSI。</p></section>'
    page+='<section id="classes"><h2>当前全部1737视野的实际标签分布</h2><p>逐文件核验SHA、尺寸、ID及像素计数；占比含背景/忽略。重复或重叠视野不是独立组织面积，表中为输入像素暴露量。</p>'+table(['域/划分','ID','组织','包含该类视野数','像素占比','像素数'],class_rows)+'</section>'
    page+='<section id="gallery"><h2>随机抽查：30个样本</h2><p>原历史patch10张（5 train＋5 val），当前CRC10张、TIGER10张（各5＋5）。两种CRC只是不同采样表现；抽样seed20261104，不按图像或模型得分挑选。当前20张与上一页面相同，原test像素未读取。点击放大，可查看来源坐标。</p><select id="group" onchange="filter()"><option value="all">全部来源</option><option value="original">原主队列历史patch</option><option value="crc">当前CRC</option><option value="tiger">TIGER</option></select><select id="split" onchange="filter()"><option value="all">全部划分</option><option>train</option><option>val</option></select></section>'+''.join(cards)
    page+='<section id="spatial"><h2>实际4×输入空间示意</h2><p class="notice">'+warning+'</p><p>蓝=train，橙=val；每张图按共同物理坐标缩放，仅显示已采样矩形，不是组织全貌。鼠标停留显示视野ID。当前集合通过不代表全候选重建已通过，跨模态、重扫描和额外细胞crop范围另审计。</p>'+''.join(maps)+'</section>'
    page+='<section id="lineage"><h2>原主队列200张来源覆盖</h2><p>历史patch数与当前空间候选/已选视野分别记账，不把两份表现形式叠加。原test只查库存清单，不纳入训练。</p>'+lineage_html+'</section><section><h2>后续数据整合的准入条件</h2><p>一个CRC源库存＋TIGER独立域；按最大实际输入范围完成空间划分后再抽样，保留分域标签和分域报告。现阶段仅生成审核清单，未替换训练数据、扩大预算或恢复实验。</p><p>原文0.8425为本图5+1点global Dice；当前0.3557为远端类别宏Dice，local为0.5471。新旧采样、提示、模型、训练预算均有变化，不能只归因某一个数据集。</p><a href="summary.json">完整审计JSON</a><a href="cohort200_lineage.csv">200张来源表CSV</a><a href="current_fields.jsonl">1737字段清单</a><a href="overlap_audit.json">空间审计</a><a href="selection.json">随机抽样清单</a></section>'
    page+='''<dialog id="viewer"><button onclick="document.getElementById('viewer').close()">关闭</button><br><img id="large"></dialog><script>function zoom(e){document.getElementById('large').src=e.src;document.getElementById('viewer').showModal()}function filter(){let g=document.getElementById('group').value,s=document.getElementById('split').value;document.querySelectorAll('.sample').forEach(e=>e.hidden=!((g==='all'||g===e.dataset.group)&&(s==='all'||s===e.dataset.split)))}</script></html>'''
    (a.output/'index.html').write_text(page)
    assert len(cards)==30 and len(image_audit)==30 and len(lineage)==200 and len(current)==1737
    dump(a.output/'acceptance.json',dict(status='accepted_readonly_inventory_and_visualization',training_approved=False,
                                        labels_checked=1737,samples=30,HTML_sha256=sha(a.output/'index.html'),summary_sha256=sha(a.output/'summary.json'),
                                        source_sha256=sha(__file__),seconds=time.monotonic()-started))
    print(json.dumps(dict(stage='complete',summary={k:v for k,v in summary.items() if k not in ['class_summary','source_hashes']},seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
