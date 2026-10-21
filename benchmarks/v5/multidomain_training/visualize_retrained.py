"""Same locked twenty cases and clicks: archived C3, retrained best and final."""
import argparse
import copy
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import time

import pyvips
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts
from benchmarks.v5.multidomain_data.prepare import dump,sha
from .episodes import EpisodeCorpus
from .evaluate import summarize
from .model import load_model
from .visualize_random import paint,configure_font,class_name,score_title,KIND_NAMES,ROOT

NAMES=dict(old='旧C3：相同新点击',best='重训14000：既定综合最佳',final='重训20000：末轮')


def checked_counts(probability,truth,valid,expected):
    counts={k:int(v[0]) for k,v in binary_counts(probability>=.5,truth,valid).items()}
    if any(expected[k]!=v for k,v in counts.items()):raise ValueError(f'count replay mismatch: {counts} vs {expected}')


def panels(data,plan,rows,overview=False):
    q,t,v=data['query_rgb'],data['truth'],data['valid'];local=plan['kind']=='local'
    support=('支持图：3正点＋2负点',data['support_rgb'],data['support_truth'],data['support_valid'],'raw',None,True)
    gt=('查询目标GT'+('：无该目标' if not (t&v).any() else ''),q,t,v,'gt',None,local)
    predictions=[(NAMES[name]+'\n'+score_title(rows[name]),q,t,v,'prediction',data[name+'_probability'],local) for name in NAMES]
    errors=('14000错误：红=误报；蓝=漏分',q,t,v,'errors',data['best_probability'],local)
    if overview:return [support,gt,*predictions,errors]
    return [support,('支持目标GT与点击',data['support_rgb'],data['support_truth'],data['support_valid'],'gt',None,True),
            ('查询原图'+('：本地点标注' if local else '：远端，无查询点击'),q,t,v,'raw',None,local),gt,*predictions,errors]


def draw(ax,panel,plan,fontsize=10):
    title,rgb,t,v,kind,prob,points=panel
    paint(ax,rgb,t,v,plan,kind,prob,points);ax.set_title(title,fontsize=fontsize)


def render_case(folder,item):
    with np.load(folder/'arrays.npz') as data:
        plan=item['plan'];fig,axes=plt.subplots(2,4,figsize=(18,10))
        for ax,panel in zip(axes.flat,panels(data,plan,item['rows'])):draw(ax,panel,plan)
        fig.suptitle(f"原随机例 #{item['case']:02d} | {plan['dataset'].upper()} | {class_name(plan)} | {KIND_NAMES[plan['kind']]}\n"
                     f"{plan['protocol']}；{plan['support_organ']} → {plan['query_organ']}；同图、同新点击、阈值0.5",fontsize=14)
        fig.text(.5,.025,'蓝＋=正点；紫×=负点；绿虚线/填充=GT；橙线/填充=预测；灰=忽略区。远端点击仅在支持图。\n'
                 '原20例不换图、不挑成功例；固定第1组GT模拟点，3正2负；开发验证/patch级，非全WSI或未见癌种评估。',ha='center',fontsize=10)
        fig.subplots_adjust(left=.012,right=.99,top=.87,bottom=.08,wspace=.10,hspace=.25)
        fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig)


def finish(output,items,full_metrics,smoke):
    if not smoke:
        for kind,title in KIND_NAMES.items():
            subset=[i for i in items if i['plan']['kind']==kind];fig,axes=plt.subplots(4,6,figsize=(24,17))
            if len(subset)!=4:raise ValueError('overview stratum incomplete')
            for row,item in enumerate(subset):
                with np.load(output/f"case_{item['case']:02d}"/'arrays.npz') as data:
                    for col,panel in enumerate(panels(data,item['plan'],item['rows'],overview=True)):
                        draw(axes[row,col],panel,item['plan'],9)
                    axes[row,0].set_title(f"#{item['case']:02d} {item['plan']['dataset'].upper()} {class_name(item['plan'])}\n支持GT模拟点击",fontsize=9)
            fig.suptitle('同一新点击的三模型对比：'+title+'\n原固定随机20例，含查询目标GT；远端点击只在支持图',fontsize=15)
            fig.text(.5,.015,'蓝＋=正点；紫×=负点；绿=GT；橙=预测；错误图红=误报、蓝=漏分。均为patch级开发验证。',ha='center',fontsize=11)
            fig.subplots_adjust(left=.008,right=.995,top=.92,bottom=.04,wspace=.08,hspace=.22)
            fig.savefig(output/f'overview_{kind}.png',dpi=140);plt.close(fig)
    cards=[];flat=[]
    for item in items:
        p=item['plan'];rows=item['rows'];title=f"#{item['case']:02d} {p['dataset'].upper()} / {class_name(p)} / {KIND_NAMES[p['kind']]}"
        old,new=rows['old'],rows['best'];delta=new['dice']-old['dice'] if new['positive'] else old['false_positive_fraction']-new['false_positive_fraction']
        change='better' if delta>1e-8 else 'worse' if delta < -1e-8 else 'same'
        cards.append(f'<article data-source="{p["dataset"]}" data-kind="{p["kind"]}" data-change="{change}"><h2>{html.escape(title)}</h2><p>'+
                     '　→　'.join(html.escape(NAMES[k]+'：'+score_title(rows[k])) for k in NAMES)+
                     f'</p><a target="_blank" href="case_{item["case"]:02d}/comparison.png"><img loading="lazy" src="case_{item["case"]:02d}/comparison.png"></a></article>')
        for name,row in rows.items():flat.append(dict(case=item['case'],model=name,dataset=p['dataset'],kind=p['kind'],target=p['target'],number=p['number'],
                                                     dice=row['dice'],bf1_um2=row['bf1_um2'],false_positive_fraction=row['false_positive_fraction']))
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    controls='<select id="source"><option value="">全部来源</option>'+''.join(f'<option>{k}</option>' for k in ['crc','tiger','ignite','ocelot'])+'</select>'
    controls+='<select id="kind"><option value="">全部任务</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in KIND_NAMES.items())+'</select>'
    controls+='<select id="change"><option value="">所有变化</option><option value="better">14000改善</option><option value="worse">14000退步</option><option value="same">相同</option></select>'
    full='；'.join(f"{NAMES[k]} Dice={v['dice']:.3f}，无目标误报={100*v['absent_false_positive_fraction']:.2f}%" for k,v in full_metrics.items())
    overviews='' if smoke else '<p>任务概览：'+'　'.join(f'<a target="_blank" href="overview_{k}.png">{v}</a>' for k,v in KIND_NAMES.items())+'</p>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>重训三模型同例对比</title><style>body{font-family:system-ui;margin:20px;background:#f4f6f8;color:#182230}header{background:white;padding:16px}nav{position:sticky;top:0;background:white;padding:12px;z-index:5}select{font-size:16px;margin-right:14px;padding:6px}article{background:white;padding:16px;margin:24px 0}img{width:100%}a{color:#175bd0}</style><header><h1>代表性点击重训：旧C3 / 14000 / 20000</h1><p>原四源×五任务随机20例，例序不变，三模型共享固定新点击（3正2负）。支持图与查询GT完整展示，点击图片可查看大图。</p><p>完整960开发验证（非下列20例均值）：'''+html.escape(full)+'''</p><p>14000按预设Dice−0.25×无目标误报面积选出；查询GT仅用于评分展示。均为patch级已有域开发验证，非全WSI或未见癌种评估。</p>'''+overviews+'</header><nav>'+controls+'<span id="count"></span></nav>'+''.join(cards)+'''<script>const ids=['source','kind','change'];function filter(){let n=0;document.querySelectorAll('article').forEach(x=>{const show=ids.every(k=>!document.getElementById(k).value||x.dataset[k]===document.getElementById(k).value);x.hidden=!show;n+=show});document.getElementById('count').textContent='显示 '+n+' 例'}ids.forEach(k=>document.getElementById(k).onchange=filter);filter();</script></html>'''
    (output/'index.html').write_text(page);dump(output/'cases.json',items)
    dump(output/'summary.json',dict(cases=len(items),sample_metrics={name:summarize([i['rows'][name] for i in items]) for name in NAMES},
        full960_metrics=full_metrics,all_replay_counts_exact=True,saved_counts_exact=True,case_selection_unchanged=True,query_GT_in_model=False,puma_used=False,test_used=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--previous-gallery',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    if ROOT not in a.output.resolve().parents:raise ValueError('output outside artifact root')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);configure_font()
    try:
        run=json.loads((a.training_run/'run.json').read_text());manifest=a.training_run/'validation_plans.json'
        if sha(manifest.read_bytes())!=run['validation_plans_sha256']:raise ValueError('validation changed')
        plans={(p['number'],p['quality_condition']):p for p in json.loads(manifest.read_text())}
        selected=json.loads((a.previous_gallery/'cases.json').read_text());dump(a.output/'selection.json',selected)
        if len(selected)!=20 or len({i['case'] for i in selected})!=20:raise ValueError('locked gallery changed')
        if a.smoke:selected=[i for i in selected if i['new_plan']['kind']=='local']
        corpus=EpisodeCorpus(Path(run['args']['catalog']),run['args']['seed'],prompt_policy='representative')
        if sha((corpus.catalog/'tiles.parquet').read_bytes())!=run['catalog_sha256']:raise ValueError('catalog changed')
        model,weights=load_model('cuda:0')
        if weights!=run['weights']:raise ValueError('initialization changed')
        model.eval().requires_grad_(False);references={};checkpoints={};full_metrics={};models={}
        best=json.loads((a.training_run/'best_checkpoint.json').read_text())
        if best['step']!=14000:raise ValueError('this gallery expects locked best step14000')
        for name,step in [('best',14000),('final',20000)]:
            checkpoint=a.training_run/f'checkpoint_step_{step:05d}.pth'
            references[name]={(r['number'],r['quality_condition']):r for r in json.loads((a.training_run/f'validation_step_{step:05d}'/'rows.json').read_text())}
            checkpoints[name]=dict(path=str(checkpoint),sha256=sha(checkpoint.read_bytes()))
            full_metrics[name]=json.loads((a.training_run/f'validation_step_{step:05d}'/'summary.json').read_text())['overall']
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['mapper']!='pixel' or saved['step']!=step:raise ValueError('checkpoint contract changed')
            candidate=copy.deepcopy(model);candidate.load_state_dict(saved['model'],strict=True);models[name]=candidate
            del saved
        del model
        # Fixed historical comparator already rerun on exactly these new clicks.
        old_root=Path('/nfs-medical3/zyh/fewclick_v2/multidomain_training/representative20k_1021_230134')
        full_metrics={'old':json.loads((old_root/'selection.json').read_text())['same_new_click_old_C3_reference'],**full_metrics}
        dump(a.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),source_sha256=sha(Path(__file__).read_bytes()),
             training_run=str(a.training_run),previous_gallery=str(a.previous_gallery),checkpoints=checkpoints,threshold=.5,smoke=a.smoke,amp=not run['args']['no_amp']))
        items=[]
        for item in selected:
            plan=item['new_plan'];key=plan['number'],plan['quality_condition']
            if plan!=plans[key]:raise ValueError('case clicks/identity differ from formal validation')
            inputs,sup=corpus.materialize(plan,'cuda:0')
            with np.load(item['source_arrays']) as archived:
                arrays={k:archived[k].copy() for k in ('support_rgb','query_rgb','truth','valid','support_truth','support_valid')}
                arrays['old_probability']=archived['probabilities'][1].copy()
            for k in ('truth','valid','support_truth','support_valid'):
                if not np.array_equal(arrays[k],sup[k].cpu().numpy()[0]):raise ValueError('archived GT differs from loader')
            for role in ('support','query'):
                source=corpus.rows[corpus.index[plan[role+'_id']]];variant=json.loads(source['variants_json'])[plan[role+'_quality']]
                with Image.open(variant['image']) as im:
                    if not np.array_equal(arrays[role+'_rgb'],np.asarray(im.convert('RGB'))):raise ValueError('archived RGB differs from loader')
            rows={'old':item['rows']['group0']}
            checked_counts(torch.from_numpy(arrays['old_probability'])[None],sup['truth'].cpu(),sup['valid'].cpu(),rows['old'])
            for name in ('best','final'):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not run['args']['no_amp']):result=models[name](**inputs,with_support=False)
                prob=result['query']['pixel_probability'].float()
                if not torch.isfinite(prob).all():raise FloatingPointError('nonfinite prediction')
                if not inputs['local'] and result['query_prompt_maps'].count_nonzero():raise ValueError('remote query contains source-coordinate clicks')
                row=references[name][key]
                if any(row[k]!=v for k,v in plan.items()):raise ValueError('reference plan changed')
                checked_counts(prob,sup['truth'],sup['valid'],row);rows[name]=row;arrays[name+'_probability']=prob.cpu().numpy()[0]
            folder=a.output/f"case_{item['case']:02d}";folder.mkdir();np.savez_compressed(folder/'arrays.npz',**arrays)
            with np.load(folder/'arrays.npz') as saved:
                for name in NAMES:checked_counts(torch.from_numpy(saved[name+'_probability'])[None],sup['truth'].cpu(),sup['valid'].cpu(),rows[name])
            entry=dict(case=item['case'],plan=plan,rows=rows,old_array_source=item['source_arrays']);dump(folder/'episode.json',entry);render_case(folder,entry);items.append(entry)
            dump(a.output/'status.json',dict(status='rendering',worker_pid=os.getpid(),completed=len(items),total=len(selected)));print('CASE',item['case'],{k:score_title(v) for k,v in rows.items()},flush=True)
        finish(a.output,items,full_metrics,a.smoke)
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),cases=len(items),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
