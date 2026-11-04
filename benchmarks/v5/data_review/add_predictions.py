"""Add independently rescored accepted predictions to the source-review website."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.v5.data_review.gallery import NAMES
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS
from benchmarks.v5.fair_comparison.metrics import score

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--review',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(exist_ok=False);assets=out/'assets';assets.mkdir()
    for p in (args.review/'assets').iterdir():(assets/p.name).symlink_to(p)
    for name in ['summary.json','acceptance.json','cohort200_lineage.csv','current_fields.jsonl','overlap_audit.json','selection.json','image_audit.json']:
        (out/name).symlink_to(args.review/name)
    sources=dict(control=BASE/'p3_support_match_1103_184618/control',match=BASE/'p3_support_match_1103_184618/match',sam=BASE/'sam_prior_repair_1104_020153/sam_native')
    allrows={};packed={};proofs={}
    for name,folder in sources.items():
        proof=json.loads((folder/'independent_acceptance.json').read_text());assert proof['status']=='accepted_P3_spatial_evaluation' and proof['rows']==3711 and not proof['smoke']
        for filename,key in [('rows.json','rows_sha256'),('masks.npz','masks_sha256'),('report.json','report_sha256')]:assert sha(folder/filename)==proof[key]
        allrows[name]=json.loads((folder/'rows.json').read_text());packed[name]=np.load(folder/'masks.npz')['masks']
        assert len(allrows[name])==len(packed[name])==3711
        proofs[name]=dict(source=str(folder),receipt_sha256=sha(folder/'independent_acceptance.json'),**proof)
    selected=json.loads((args.review/'selection.json').read_text());ids={r['tile_id'] for r in selected['current']}
    reference=allrows['match'];indices=[i for i,r in enumerate(reference) if r['budget']=='3+2' and r['plan']['query_id'] in ids]
    rng=random.Random(20261104)
    for domain in ['crc','tiger']:
        chosen=[i for i in indices if reference[i]['scope']=='local' and reference[i]['plan']['dataset']==domain]
        remaining=[i for i,r in enumerate(reference) if i not in indices and r['scope']=='local' and r['budget']=='3+2' and r['plan']['dataset']==domain]
        if len(chosen)<5:indices+=rng.sample(remaining,5-len(chosen))
    indices=sorted(indices,key=lambda i:(reference[i]['scope'],reference[i]['plan']['dataset'],i))
    # Selection uses task identity and scope, never stored scores.
    (out/'prediction_selection.json').write_text(json.dumps(dict(seed=20261104,indices=indices,policy='all existing 3+2 tasks querying fixed20 review samples; local padded to five per domain with fixed random sampling, no score filtering'),indent=2))
    _,catalog=load_cell_catalog(TASKS);records=[];cards=[]
    def rgb(identity):
        row=catalog[identity];v=json.loads(row['variants_json'])['original'];assert sha(v['image'])==v['image_sha256']
        return Image.open(v['image']).convert('RGB')
    for case,index in enumerate(indices):
        reference_row=reference[index];plan=reference_row['plan'];scope=reference_row['scope'];domain=plan['dataset'];q=catalog[plan['query_id']]
        query=rgb(plan['query_id']);source=rgb(plan['support_id']);gt=np.asarray(Image.open(q['labels']))
        assert hashlib.sha256(gt.tobytes()).hexdigest()==q['labels_array_sha256'];valid=gt!=255;truth=(gt==plan['target'])&valid
        assert q['split']=='val'
        if scope=='remote':assert plan['support_id']!=plan['query_id']
        drawn=source.copy();draw=ImageDraw.Draw(drawn)
        for sign,color in [('positive',(0,255,70)),('negative',(255,0,210))]:
            for x,y in plan[sign+'_xy']:
                x,y=x*512-.5,y*512-.5;draw.ellipse([x-6,y-6,x+6,y+6],outline=color,width=3)
        assert len(plan['positive_xy'])==3 and len(plan['negative_xy'])==2
        def overlay(mask,color):
            image=np.asarray(query).copy();image[mask]=(image[mask]*.55+np.asarray(color)*.45).astype(np.uint8);return Image.fromarray(image)
        figures=[]
        def add(image,caption,key):
            name=f'prediction_{case:03}_{key}.png';image.save(assets/name)
            assert np.array_equal(np.asarray(Image.open(assets/name)),np.asarray(image))
            figures.append(f'<figure><img loading="lazy" src="assets/{name}" onclick="zoom(this)"><figcaption>{html.escape(caption)}</figcaption></figure>')
        add(drawn,'支持图：绿色正点、粉色负点','support');add(query,'查询原图'+('（本图接收这些点）' if scope=='local' else '（无人工点）'),'query')
        add(overlay(truth,[255,210,0]),'目标组织 GT（黄）','gt')
        metrics={};masks={}
        for name,title in [('control','FewClick 对照'),('match','FewClick＋匹配模块'),('sam','SAM 当前适配方案')]:
            row=allrows[name][index];assert {k:row[k] for k in ['scope','budget','plan']}=={k:reference_row[k] for k in ['scope','budget','plan']}
            mask=np.unpackbits(packed[name][index],count=512*512).reshape(512,512).astype(bool)
            mpp=json.loads(q['input_mpp_xy_json']);actual=score(mask,gt==plan['target'],valid,mpp);assert actual==row['metrics']
            metrics[name]=actual;masks[name]=mask;d=actual['dice'];caption=title+(' · 无目标，Dice不适用' if d is None else f' · Dice {d:.3f}')
            caption+=f" · FP/有效面积 {actual['false_positive_fraction']:.1%}"
            add(overlay(mask&valid,[255,65,65]),caption,name)
            Image.fromarray((mask*255).astype(np.uint8)).save(assets/f'prediction_{case:03}_{name}_binary.png')
        mask=masks['match'];error=np.asarray(query).copy();error[mask&truth]=[30,210,50];error[mask&~truth&valid]=[255,40,40];error[~mask&truth]=[30,90,255];error[~valid]=[50,50,50]
        add(Image.fromarray(error),'匹配模型误差：绿=命中，红=误检，蓝=漏检，灰=忽略','error')
        kind='本图交互' if scope=='local' else '远端迁移';present='present' if truth.any() else 'absent'
        metadata=dict(case=case,row_index=index,scope=scope,plan=plan,metrics=metrics,query_split=q['split'],
                      query_source=q['source_id'],query_in_fixed_review=plan['query_id'] in ids,
                      masks_sha256={name:hashlib.sha256(masks[name].tobytes()).hexdigest() for name in masks})
        records.append(metadata)
        title=f'{domain.upper()} · {kind} · {NAMES[domain][plan["target"]]} · '+('目标存在' if truth.any() else '目标缺失')
        cards.append(f'<article class="prediction" data-domain="{domain}" data-scope="{scope}" data-present="{present}"><h3>{html.escape(title)}</h3><p>3正2负 · 查询来自验证集 · '+('属于上方固定随机样本' if plan['query_id'] in ids else '补充随机本图任务')+'</p><div class="images">'+''.join(figures)+'</div><details><summary>任务、原始指标及二值预测下载</summary><p>'+''.join(f'<a href="assets/prediction_{case:03}_{n}_binary.png">{n} 二值mask</a>' for n in masks)+'</p><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></details></article>')
        print(json.dumps(dict(stage='rendering',cases=case+1,total=len(indices))),flush=True)
    page=(args.review/'index.html').read_text()
    counts=Counter(r['scope']+'|'+r['plan']['dataset'] for r in records)
    matched={r['plan']['query_id'] for r in records if r['query_in_fixed_review']};missing=[s for s in selected['current'] if s['tile_id'] not in matched]
    intro='<section id="predictions"><h2>推理结果：本图交互与远端迁移</h2><p>复用已独立验收的首seed固定末步模型，未新增训练或GPU推理。三种方法接收同一个任务和支持点击；远端查询人工点为0，支持mask不作为推理输入。SAM为当前专门实现的支持提示适配方案，不代表SAM的最优能力。</p><p>模型为本次CVPR探索的重新训练版本，不是原论文0.8成绩对应的checkpoint。三模型各1000×4预算，公共预训练、编码器分辨率、容量及部分正则不同；本页是已完成实验的逐例对照，不是充分调优的最终排名。</p><p>优先展示原固定20个样本中已有3+2任务的全部预测，不按Dice挑选；每个域本图不足5例时固定随机补齐。'+html.escape(str(dict(counts)))+'</p><p class="notice">原主队列的10个历史patch和当前10个train样本没有这组已验收验证预测；另有部分val图无既有任务，本页明确保留缺口。未把别的图的结果贴到这些图上。原固定20样本中有预测的不同查询图 '+str(len(matched))+' 张，无既有预测 '+str(len(missing))+' 张。各查询可包含多个目标和支持来源，不是独立样本数。</p><p>红色覆盖层为预测前景，原图仍可对照；二值mask可在任务详情下载。无目标任务单列误报，不把空目标Dice设为1。</p><select id="pred_scope" onchange="filterPred()"><option value="all">全部任务</option><option value="local">本图交互</option><option value="remote">远端迁移</option></select><select id="pred_domain" onchange="filterPred()"><option value="all">全部数据域</option><option value="crc">CRC</option><option value="tiger">TIGER</option></select><select id="pred_present" onchange="filterPred()"><option value="all">存在/缺失全部</option><option value="present">目标存在</option><option value="absent">目标缺失</option></select></section>'
    page=page.replace('<a href="#overview">','<a href="#predictions">推理对照</a><a href="#overview">',1)
    page=page.replace('<section id="gallery">',intro+''.join(cards)+'<section id="gallery">',1)
    page=page.replace('</body>','') # Original generated document has no explicit body end.
    page=page.replace('</html>','''<script>function filterPred(){let s=document.getElementById('pred_scope').value,d=document.getElementById('pred_domain').value,p=document.getElementById('pred_present').value;document.querySelectorAll('.prediction').forEach(e=>e.hidden=!((s==='all'||e.dataset.scope===s)&&(d==='all'||e.dataset.domain===d)&&(p==='all'||e.dataset.present===p)))}</script></html>''')
    (out/'index.html').write_text(page)
    result=dict(status='accepted_replayed_prediction_visualization',cases=len(records),counts=dict(counts),records=records,
                source_proofs=proofs,seed=20261103,fixed_review_query_coverage=len(matched),fixed_review_missing=missing,
                historical_samples_have_no_this_checkpoint_predictions=True,independent_mask_rescores=len(records)*3,
                training_paused=True,new_GPU_inference=False,source_sha256=sha(__file__),original_review=str(args.review))
    (out/'predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    assert len(records)==35 and sum(r['scope']=='local' for r in records)==10
    print(json.dumps(dict(status='complete',cases=len(records),mask_rescores=len(records)*3,coverage=len(matched))))


if __name__=='__main__':main()
