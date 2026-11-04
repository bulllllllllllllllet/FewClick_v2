"""CPU-only reproducible dataset review; never selects examples by model score."""
import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS

NAMES = {'crc': ['肿瘤上皮','肿瘤间质','背景','坏死','正常腺体','正常间质','黏膜下层/浆膜','肌层','淋巴聚集','黏液','脂肪','血液'],
         'tiger': ['背景/未标组织(按标签定义)','浸润癌','肿瘤相关间质','原位癌','健康腺体','坏死','炎性间质','其他复合组织']}
COLORS = np.array([[230,70,70],[65,150,215],[210,210,210],[140,75,160],[240,155,45],[55,165,120],
                   [150,125,85],[235,100,175],[80,100,200],[80,205,205],[220,215,70],[160,45,60]], dtype=np.uint8)


def encoded(image):
    stream = io.BytesIO(); image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def figure(image, title):
    return '<figure><img loading="lazy" src="'+encoded(image)+'"><figcaption>'+html.escape(title)+'</figcaption></figure>'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261104); a = p.parse_args(); a.output.mkdir(exist_ok=False)
    _, rows = load_cell_catalog(TASKS)
    rng = random.Random(a.seed); selected = []
    for dataset in ['crc', 'tiger']:
        for split in ['train', 'val']:
            pool = sorted([r for r in rows.values() if r['dataset']==dataset and r['split']==split], key=lambda r:r['tile_id'])
            selected.extend(rng.sample(pool, 5))
    # Freeze identities before opening any image, label or result.
    selection = dict(seed=a.seed,policy='uniform without replacement within each dataset/split; five train and five val',
                     tiles=[dict(dataset=r['dataset'],split=r['split'],tile_id=r['tile_id']) for r in selected])
    (a.output/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    cache = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_repair_rebalance_1102_210552/normalized')
    entries = {r['tile_id']:r for r in map(json.loads,(cache/'index.jsonl').read_text().splitlines())}
    blocks=[]; audit=[]
    for r in selected:
        ds=r['dataset']; variant=json.loads(r['variants_json'])['original']; rawpath=Path(entries[r['tile_id']]['file'])
        assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==entries[r['tile_id']]['sha256']
        with np.load(rawpath) as f: images=[Image.fromarray(f['image_'+str(i)]) for i in range(3)]
        gt=np.asarray(Image.open(r['labels'])); assert hashlib.sha256(gt.tobytes()).hexdigest()==r['labels_array_sha256']
        original=Image.open(variant['image']).convert('RGB'); assert np.array_equal(np.asarray(original),np.asarray(images[0]))
        assert images[0].size==gt.shape[::-1]==(512,512)
        assert set(np.unique(gt)).issubset(set(range(len(NAMES[ds])))|{255})
        colors=np.zeros((*gt.shape,3),np.uint8)
        for label in np.unique(gt): colors[gt==label]=[40,40,40] if label==255 else COLORS[int(label)]
        labelimg=Image.fromarray(colors); overlay=Image.blend(images[0],labelimg,.4)
        values,counts=np.unique(gt,return_counts=True)
        labels=' '.join('<span><i style="background:rgb('+','.join(map(str,([40,40,40] if int(v)==255 else COLORS[int(v)])))+')"></i>'+html.escape('255 忽略' if v==255 else f'{v} {NAMES[ds][int(v)]}')+f' {n/gt.size:.1%}</span>' for v,n in zip(values,counts))
        metadata=dict(tile_id=r['tile_id'],dataset=ds,split=r['split'],source_id=r['source_id'],space_key=r['space_key'],
                      mpp=json.loads(r['input_mpp_xy_json']),counts={str(v):int(n) for v,n in zip(values,counts)},
                      image_sha256=hashlib.sha256(Path(variant['image']).read_bytes()).hexdigest(),
                      labels_array_sha256=r['labels_array_sha256'],model_fine_equals_original=True)
        audit.append(metadata)
        blocks.append(f'<article class="sample" data-dataset="{ds}" data-split="{r["split"]}"><h3>{ds.upper()} · {r["split"]} · {r["tile_id"]}</h3><div class="grid">'+figure(images[0],'实际模型 fine 输入')+figure(labelimg,'完整多类组织标注')+figure(overlay,'原图＋标注叠加')+'</div><div class="legend">'+labels+'</div><details><summary>2× / 4× 上下文与来源</summary><div class="grid">'+figure(images[1],'实际 2× 上下文输入')+figure(images[2],'实际 4× 上下文输入')+'</div><pre>'+html.escape(json.dumps(metadata,ensure_ascii=False,indent=2))+'</pre></details></article>')
    study=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p3_support_match_1103_184618/match')
    result=json.loads((study/'rows.json').read_text()); masks=np.load(study/'masks.npz')['masks']; predictions=[]
    for ds in ['crc','tiger']:
        pool=[i for i,r in enumerate(result) if r['scope']=='remote' and r['budget']=='3+2' and r['plan']['dataset']==ds]
        for index in rng.sample(pool,5):
            row=result[index];plan=row['plan'];q=rows[plan['query_id']];s=rows[plan['support_id']]
            def rgb(r):return Image.open(json.loads(r['variants_json'])['original']['image']).convert('RGB')
            support=rgb(s);draw=ImageDraw.Draw(support)
            for sign,c in [('positive',(20,255,30)),('negative',(255,30,210))]:
                for x,y in plan[sign+'_xy']:
                    x,y=x*512-.5,y*512-.5;draw.ellipse((x-6,y-6,x+6,y+6),outline=c,width=3)
            query=rgb(q);gt=np.asarray(Image.open(q['labels']));mask=np.unpackbits(masks[index],count=512*512).reshape(512,512).astype(bool);truth=gt==plan['target']
            def binary_overlay(m,color):
                arr=np.asarray(query).copy();arr[m]=(arr[m]*.5+np.array(color)*.5).astype(np.uint8);return Image.fromarray(arr)
            error=np.asarray(query).copy();error[mask&truth]=[40,200,60];error[mask&~truth&(gt!=255)]=[255,50,50];error[~mask&truth]=[40,100,255];error[gt==255]=[50,50,50]
            score=row['metrics']['dice'];fpa=row['metrics']['false_positive_fraction']
            title=f'{ds.upper()} · 远端 · {NAMES[ds][plan["target"]]} · '+('无目标，Dice不适用' if score is None else f'Dice {score:.3f}')+f' · FP/有效面积 {fpa:.1%}'
            blocks.append('<article><h3>'+html.escape(title)+'</h3><div class="grid">'+figure(support,'支持图：绿=正点，粉=负点')+figure(query,'查询原图：人工点为0')+figure(binary_overlay(truth,[255,215,40]),'目标 GT（黄）')+figure(binary_overlay(mask,[255,80,80]),'匹配模型预测（红）')+figure(Image.fromarray(error),'误差：绿TP / 红FP / 蓝FN / 灰ignore')+'</div><p>'+html.escape(f'support={plan["support_id"]}; query={plan["query_id"]}; '+plan['kind'])+'</p></article>')
            predictions.append(dict(row_index=index,plan=plan,metrics=row['metrics']))
    counts=Counter(r['dataset']+'|'+r['split'] for r in rows.values())
    text='''<!DOCTYPE html><html lang="zh"><meta charset="UTF-8"><meta name="viewport" content="width=device-width"><title>FewClick 数据检查</title><style>body{font:16px system-ui;margin:24px;background:#f3f5f8;color:#172235}h1{font-size:27px}article{background:white;border-radius:12px;padding:18px;margin:20px 0;box-shadow:0 2px 7px #ddd}h3{overflow-wrap:anywhere}.grid{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;flex:1;min-width:220px;max-width:512px}img{width:100%;cursor:zoom-in}figcaption{padding:6px;font-size:14px}.legend{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}i{display:inline-block;width:16px;height:16px;vertical-align:middle;margin-right:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}button,select{font-size:16px;padding:8px;margin:6px}header{background:#fff;padding:18px;border-radius:12px}dialog{max-width:95vw;max-height:95vh}dialog img{width:auto;max-width:90vw;max-height:85vh}</style><header><h1>FewClick 当前数据集抽查</h1><p>CRC / TIGER 各10张：各随机5张train＋5张val。抽样seed=20261104，先锁定身份再读取图像/标签；不按质量或得分挑选。共20个数据样本，另附每域5个随机远端预测任务。255为忽略标签，CRC背景为类别2；完整标签不等于模型推理输入，模型只接收图像和允许的支持提示。</p><p>当前库存：'''+html.escape(str(dict(counts)))+'''。这是同队列空间块划分，保留来源；不是患者独立测试。其他历史数据集未用于本次训练，不混入当前库存。</p><p>原文0.8425是5+1点的patch全局像素Dice；当前远端类别宏Dice0.3557、本图任务平均Dice0.5471，协议、训练和统计不同。当前训练为1000步×4任务的小试，不是原论文模型的复现。</p><label>数据集 <select id="dataset"><option value="all">全部</option><option>crc</option><option>tiger</option></select></label><label>划分 <select id="split"><option value="all">全部</option><option>train</option><option>val</option></select></label><p>点击图片放大；展开卡片查看多尺度图像和来源。随机样本不保证覆盖所有类别。</p></header>'''+''.join(blocks)+'''<dialog id="zoom"><button onclick="this.parentNode.close()">关闭</button><br><img id="large"></dialog><script>document.querySelectorAll('article img').forEach(e=>e.onclick=()=>{document.getElementById('large').src=e.src;document.getElementById('zoom').showModal()});function filter(){let d=document.getElementById('dataset').value,s=document.getElementById('split').value;document.querySelectorAll('.sample').forEach(e=>e.hidden=!((d==='all'||e.dataset.dataset===d)&&(s==='all'||e.dataset.split===s)))}document.querySelectorAll('select').forEach(e=>e.onchange=filter);</script></html>'''
    (a.output/'index.html').write_text(text)
    (a.output/'audit.json').write_text(json.dumps(dict(status='accepted_gallery_inputs',samples=audit,predictions=predictions,counts=dict(counts),seed=a.seed,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
    assert len(audit)==20 and len(predictions)==10
    assert text.count('<article')==30
    print(json.dumps(dict(output=str(a.output/'index.html'),samples=20,predictions=10,bytes=len(text.encode()))))


if __name__=='__main__':main()
