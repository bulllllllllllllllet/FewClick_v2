"""Visual comparison on the previously locked20 cases; fixed C3, changed clicks."""
import argparse
import html
import json
from pathlib import Path

import pyvips
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.v5.multidomain_data.prepare import dump
from .visualize_random import paint,configure_font,class_name,score_title,KIND_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--previous-gallery',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True);configure_font()
    merged=Path(json.loads((a.experiment/'merge_selection.json').read_text())['output'])
    cases={r['index']:r for r in json.loads((merged/'cases.json').read_text())}
    rows={(r['index'],r['condition']):r for r in json.loads((merged/'rows.json').read_text())}
    locations={}
    for rec in json.loads((a.experiment/'full_selection.json').read_text()):
        folder=Path(rec['output'])
        for f in (folder/'gallery_arrays').glob('*.npz'):locations[int(f.stem)]=f
    old=json.loads((a.previous_gallery/'cases.json').read_text());items=[];cards=[]
    for item in old:
        plan=item['plan'];index=next(i for i,c in cases.items() if c['legacy_plan']['number']==plan['number'] and c['legacy_plan']['quality_condition']==plan['quality_condition'])
        c=cases[index];new=c['new_plans'][0];oracle=c['new_plans'][int(c['oracle_group'][-1])];arrays=np.load(locations[index]);probs=arrays['probabilities']
        folder=a.output/f"case_{item['case']:02d}";folder.mkdir();r={k:rows[index,k] for k in ('legacy','group0','ensemble','oracle')}
        data=dict(case=item['case'],index=index,old_plan=plan,new_plan=new,oracle_plan=oracle,rows=r,source_arrays=str(locations[index]))
        dump(folder/'episode.json',data);items.append(data)
        fig,axes=plt.subplots(2,5,figsize=(21,10))
        panels=[('旧规则：支持GT与点击',arrays['support_rgb'],arrays['support_truth'],arrays['support_valid'],plan,'gt',None,True),
                ('新规则：固定第1组支持GT与点击',arrays['support_rgb'],arrays['support_truth'],arrays['support_valid'],new,'gt',None,True),
                ('五组中GT最佳的支持点（仅诊断）',arrays['support_rgb'],arrays['support_truth'],arrays['support_valid'],oracle,'gt',None,True),
                ('查询原图',arrays['query_rgb'],arrays['truth'],arrays['valid'],plan,'raw',None,False),
                ('查询目标GT',arrays['query_rgb'],arrays['truth'],arrays['valid'],plan,'gt',None,False),
                ('旧规则：同一C3\n'+score_title(r['legacy']),arrays['query_rgb'],arrays['truth'],arrays['valid'],plan,'prediction',probs[0],False),
                ('新规则固定第1组：同一C3\n'+score_title(r['group0']),arrays['query_rgb'],arrays['truth'],arrays['valid'],new,'prediction',probs[1],False),
                ('新规则5组平均概率\n'+score_title(r['ensemble']),arrays['query_rgb'],arrays['truth'],arrays['valid'],new,'prediction',probs[6],False),
                ('新规则5组GT最佳（仅诊断上限）\n'+score_title(r['oracle']),arrays['query_rgb'],arrays['truth'],arrays['valid'],oracle,'prediction',probs[1+int(c['oracle_group'][-1])],False),
                ('新规则固定第1组错误：红误报/蓝漏分',arrays['query_rgb'],arrays['truth'],arrays['valid'],new,'errors',probs[1],False)]
        for ax,(title,rgb,t,v,p,kind,prob,points) in zip(axes.flat,panels):paint(ax,rgb,t,v,p,kind,prob,points);ax.set_title(title,fontsize=10)
        fig.suptitle(f"原随机例 #{item['case']:02d} | {plan['dataset'].upper()} | {class_name(plan)} | {KIND_NAMES[plan['kind']]}\n"
                     '同一图像/权重/阈值，只换点击；固定第一组为正式单组对照，GT最佳不是部署结果',fontsize=14)
        fig.text(.5,.025,'蓝＋=正点，紫×=负点，绿色=GT，橙色=预测，灰=忽略。每组同3正2负，5组使用更多提示。\n'
                 '原20例身份不变；新点只由支持GT模拟；查询GT仅评分与显式oracle。仍为开发val/patch级，不是新训练结果。',ha='center',fontsize=11)
        fig.subplots_adjust(left=.012,right=.99,top=.86,bottom=.08,wspace=.1,hspace=.25);fig.savefig(folder/'comparison.png',dpi=140);plt.close(fig);arrays.close()
        title=f"#{item['case']:02d} {plan['dataset'].upper()} / {class_name(plan)} / {KIND_NAMES[plan['kind']]}"
        cards.append(f'<article data-source="{plan["dataset"]}"><h2>{html.escape(title)}</h2><p>旧点：{html.escape(score_title(r["legacy"]))} → 新固定一组：{html.escape(score_title(r["group0"]))}</p><a target="_blank" href="case_{item["case"]:02d}/comparison.png"><img loading="lazy" src="case_{item["case"]:02d}/comparison.png"></a></article>')
    for kind in KIND_NAMES:
        subset=[i for i in items if i['old_plan']['kind']==kind];fig,axes=plt.subplots(4,5,figsize=(18.5,16.5))
        for n,item in enumerate(subset):
            c=cases[item['index']];old=c['legacy_plan'];new=c['new_plans'][0];data=np.load(item['source_arrays']);prob=data['probabilities']
            panels=[('旧支持GT与点',data['support_rgb'],data['support_truth'],data['support_valid'],old,'gt',None,True),
                    ('新支持GT与点',data['support_rgb'],data['support_truth'],data['support_valid'],new,'gt',None,True),
                    ('查询GT',data['query_rgb'],data['truth'],data['valid'],old,'gt',None,False),
                    ('旧点\n'+score_title(item['rows']['legacy']),data['query_rgb'],data['truth'],data['valid'],old,'prediction',prob[0],False),
                    ('新固定一组\n'+score_title(item['rows']['group0']),data['query_rgb'],data['truth'],data['valid'],new,'prediction',prob[1],False)]
            for ax,(title,rgb,t,v,p,panel,proba,points) in zip(axes[n],panels):paint(ax,rgb,t,v,p,panel,proba,points);ax.set_title(title,fontsize=9)
            axes[n,0].set_title(f"#{item['case']:02d} {old['dataset'].upper()} {class_name(old)}\n旧点",fontsize=9);data.close()
        fig.suptitle('旧点 → 代表性新点：'+KIND_NAMES[kind]+'\n同原随机样本、同C3权重，只改变支持点；不按预测效果选第一组',fontsize=15)
        fig.text(.5,.012,'蓝＋=正点，紫×=负点；绿=GT，橙=预测。全部开发val，未重新训练。',ha='center',fontsize=11)
        fig.subplots_adjust(left=.012,right=.99,top=.92,bottom=.035,wspace=.08,hspace=.22);fig.savefig(a.output/f'overview_{kind}.png',dpi=140);plt.close(fig)
    page='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>代表性点击对照</title><style>body{font-family:system-ui;background:#f4f6f8;margin:24px}header{position:sticky;top:0;background:white;padding:16px;z-index:5}article{background:white;padding:16px;margin:24px 0}img{width:100%}select{font-size:17px}</style><header><h1>同一C3：旧点与代表性新点</h1><p>原随机20例，固定第一组 / 五组平均 / GT最佳上限。没有重训，GT最佳不可作为部署分数。</p><select id="source"><option value="">全部数据源</option><option>crc</option><option>tiger</option><option>ignite</option><option>ocelot</option></select></header>'''+''.join(cards)+'''<script>document.querySelector('#source').onchange=function(){document.querySelectorAll('article').forEach(x=>x.hidden=this.value&&x.dataset.source!==this.value)}</script></html>'''
    (a.output/'index.html').write_text(page);dump(a.output/'cases.json',items);dump(a.output/'status.json',dict(status='complete',cases=len(items)));print('GALLERY_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
