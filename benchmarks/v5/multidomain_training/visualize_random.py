"""Locked stratified random gallery with support clicks, query GT and errors."""
import argparse
import copy
import html
import inspect
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
from matplotlib import font_manager, patheffects

from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_counts
from benchmarks.v5.multidomain_data.prepare import LABELS, dump, sha
from .episodes import EpisodeCorpus, DATASETS, KINDS
from .model import load_model

ROOT=Path('/nfs-medical3/zyh/fewclick_v2')
KIND_NAMES=dict(local='本地提示',same_present='同片/ROI远处：有目标',cross_present='跨片/图像：有目标',
                same_absent='同片/ROI远处：无目标',cross_absent='跨片/图像：无目标')
CRC_NAMES=['肿瘤上皮','肿瘤间质','背景','坏死','正常腺体','正常间质','黏膜下层/浆膜','肌层','淋巴聚集','黏液','脂肪','血液']
TIGER_NAMES={1:'浸润癌',2:'肿瘤相关间质',3:'原位癌',4:'健康腺体',5:'坏死',6:'炎性间质'}


def select_plans(plans, seed):
    """Select identities before reading any scores; quality pairs counted once."""
    rng=np.random.default_rng(seed);selected=[]
    for dataset in DATASETS:
        for kind in KINDS:
            pool=[p for p in plans if p['dataset']==dataset and p['kind']==kind and p['quality_condition']=='original']
            if not pool:raise ValueError(f'empty random stratum: {dataset}/{kind}')
            picked=pool[int(rng.integers(len(pool)))];selected.append(dict(case=len(selected)+1,plan=picked,pool_size=len(pool)))
    if len({(s['plan']['number'],s['plan']['quality_condition']) for s in selected})!=20:raise ValueError('duplicate sampled task')
    return selected


def class_name(plan):
    ds,target=plan['dataset'],plan['target']
    return CRC_NAMES[target] if ds=='crc' else TIGER_NAMES[target] if ds=='tiger' else LABELS[ds][target]


def configure_font():
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    font_manager.fontManager.addfont(font);plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False


def click_layer(ax, plan, size):
    height,width=size
    for sign,color,marker,prefix in [('positive','#0085FF','+','P'),('negative','#D000E8','x','N')]:
        xy=np.array(plan[sign+'_xy']);x=xy[:,0]*width-.5;y=xy[:,1]*height-.5
        dots=ax.scatter(x,y,s=125,c=color,marker=marker,linewidths=2.5,zorder=10)
        dots.set_path_effects([patheffects.withStroke(linewidth=4,foreground='white')])
        for i,(xx,yy) in enumerate(zip(x,y)):
            offsets=[(9,10),(-24,9),(8,-18),(-25,-18),(7,24)]
            ax.annotate(f'{prefix}{i+1}',(xx,yy),xytext=offsets[i%len(offsets)],textcoords='offset points',color=color,fontsize=8,
                        arrowprops=dict(arrowstyle='-',color=color,lw=.8),
                        path_effects=[patheffects.withStroke(linewidth=2,foreground='white')],zorder=11)


def paint(ax, rgb, truth, valid, plan, panel, probability=None, points=False):
    ax.imshow(rgb)
    overlay=np.zeros((*truth.shape,4),np.float32)
    if panel=='gt':overlay[truth&valid]=[0,.8,.2,.30]
    elif panel=='prediction':overlay[(probability>=.5)&valid]=[1,.5,0,.23]
    elif panel=='errors':
        prediction=probability>=.5
        overlay[prediction&truth&valid]=[0,.8,.2,.18]
        overlay[prediction&~truth&valid]=[1,.05,.02,.80]
        overlay[~prediction&truth&valid]=[0,.30,1,.82]
    ax.imshow(overlay)
    if panel in ('gt','prediction','probability'):
        if truth[valid].any() and (~truth[valid]).any():
            ax.contour(np.ma.array(truth.astype(float),mask=~valid),levels=[.5],colors=['#00D43B'],linestyles='dashed',linewidths=1)
        if panel=='prediction' and (probability[valid]>=.5).any() and (probability[valid]<.5).any():
            ax.contour(np.ma.array(probability,mask=~valid),levels=[.5],colors=['#FF8000'],linewidths=1)
    if panel=='probability':ax.imshow(np.ma.array(probability,mask=~valid),cmap='magma',vmin=0,vmax=1,alpha=.65)
    ignore=np.zeros((*truth.shape,4),np.float32);ignore[~valid]=[.43,.43,.43,.78];ax.imshow(ignore)
    if points:click_layer(ax,plan,truth.shape)
    ax.set_xlim(-.5,truth.shape[1]-.5);ax.set_ylim(truth.shape[0]-.5,-.5);ax.axis('off')


def score_title(row):
    if row['positive']==0:return f"无目标：误报面积 {100*row['false_positive_fraction']:.2f}%"
    precision=row['tp']/max(row['tp']+row['fp'],1);recall=row['tp']/max(row['tp']+row['fn'],1)
    return f"Dice {row['dice']:.3f} | 精确率 {precision:.2f} | 召回率 {recall:.2f}"


def render_case(folder, item):
    plan=item['plan'];data=np.load(folder/'arrays.npz');query=data['query_rgb'];truth=data['truth'];valid=data['valid']
    fig,axes=plt.subplots(2,4,figsize=(17,9.9))
    panels=[('支持图：3正点 + 2负点',data['support_rgb'],data['support_truth'],data['support_valid'],'raw',None,True),
            ('支持目标GT（仅用于说明/模拟点）',data['support_rgb'],data['support_truth'],data['support_valid'],'gt',None,True),
            ('查询原图'+('（本地，显示相同点）' if plan['kind']=='local' else '（远端，无查询点击）'),query,truth,valid,'raw',None,plan['kind']=='local'),
            ('查询目标GT'+('：本例无该组织' if not (truth&valid).any() else ''),query,truth,valid,'gt',None,plan['kind']=='local'),
            ('训练前 J5+dense\n'+score_title(item['initial']),query,truth,valid,'prediction',data['initial_probability'],plan['kind']=='local'),
            ('当前 C3\n'+score_title(item['C3']),query,truth,valid,'prediction',data['C3_probability'],plan['kind']=='local'),
            ('当前C3错误图：红=误分，蓝=漏分',query,truth,valid,'errors',data['C3_probability'],plan['kind']=='local'),
            ('当前C3概率：黑紫低 → 橙黄高（0–1）',query,truth,valid,'probability',data['C3_probability'],plan['kind']=='local')]
    for ax,(title,rgb,t,v,panel,prob,points) in zip(axes.flat,panels):
        paint(ax,rgb,t,v,plan,panel,prob,points);ax.set_title(title,fontsize=10)
    fig.suptitle(f"随机例 {item['case']:02d} | {plan['dataset'].upper()} | 目标：{class_name(plan)}\n"
                 f"{KIND_NAMES[plan['kind']]}；实际来源协议 {plan['protocol']}；{plan['support_organ']} → {plan['query_organ']}",fontsize=14)
    fig.text(.5,.025,'蓝＋=支持正点；紫×=支持负点；绿虚线/绿色填充=GT；橙线/橙色填充=预测；红=误报；蓝=漏分；灰=忽略区。\n'
             '固定随机分层抽样，不按效果选图；全部为开发val/GT模拟点、512×512相同完整视野，非整片WSI或未见癌种盲测。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.015,right=.985,bottom=.085,top=.86,wspace=.12,hspace=.28)
    fig.savefig(folder/'comparison.png',dpi=145);plt.close(fig);data.close()


def make_overviews(output, items):
    for kind in KINDS:
        group=[r for r in items if r['plan']['kind']==kind];fig,axes=plt.subplots(4,5,figsize=(18.5,16.8))
        for row,item in enumerate(group):
            folder=output/f"case_{item['case']:02d}";data=np.load(folder/'arrays.npz');p=item['plan'];t=data['truth'];v=data['valid']
            panels=[(data['support_rgb'],data['support_truth'],data['support_valid'],'raw',None,True),
                    (data['query_rgb'],t,v,'gt',None,p['kind']=='local'),
                    (data['query_rgb'],t,v,'prediction',data['initial_probability'],p['kind']=='local'),
                    (data['query_rgb'],t,v,'prediction',data['C3_probability'],p['kind']=='local'),
                    (data['query_rgb'],t,v,'errors',data['C3_probability'],p['kind']=='local')]
            for col,(rgb,truth,valid,panel,prob,points) in enumerate(panels):paint(axes[row,col],rgb,truth,valid,p,panel,prob,points)
            titles=[f"#{item['case']:02d} {p['dataset'].upper()}：支持图/点击\n{class_name(p)}",'查询GT（绿色）\n'+p['protocol'],
                    '训练前\n'+score_title(item['initial']),'当前C3\n'+score_title(item['C3']),'当前错误\n红=误分；蓝=漏分；灰=忽略']
            for ax,title in zip(axes[row],titles):ax.set_title(title,fontsize=9)
            data.close()
        fig.suptitle('分层随机样本：'+KIND_NAMES[kind]+'\n同一模型/阈值/点击；每源随机1例，未按效果挑选；均为开发val任务',fontsize=16)
        fig.text(.5,.015,'蓝＋=正提示；紫×=负提示。远端点击只在支持图；绿虚线=GT；橙线=预测。全部为patch级相同视野。',ha='center',fontsize=11)
        fig.subplots_adjust(left=.012,right=.99,bottom=.045,top=.925,wspace=.07,hspace=.23)
        fig.savefig(output/f'overview_{kind}.png',dpi=145);plt.close(fig)


def write_gallery(output, items):
    cards=[]
    for r in items:
        p=r['plan'];title=f"#{r['case']:02d} {p['dataset'].upper()} / {class_name(p)} / {KIND_NAMES[p['kind']]}"
        cards.append(f'<article data-source="{p["dataset"]}" data-kind="{p["kind"]}"><h2>{html.escape(title)}</h2>'
                     f'<p>训练前：{html.escape(score_title(r["initial"]))}　→　C3：{html.escape(score_title(r["C3"]))}</p>'
                     f'<a href="case_{r["case"]:02d}/comparison.png" target="_blank"><img loading="lazy" src="case_{r["case"]:02d}/comparison.png"></a></article>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FewClick C3 随机样本</title><style>body{font-family:system-ui;margin:24px;background:#f5f7fa;color:#182230}header{position:sticky;top:0;background:#fff;padding:14px;z-index:5;border-bottom:1px solid #ddd}article{background:white;margin:24px 0;padding:18px;border-radius:10px}img{width:100%;height:auto}select{font-size:16px;padding:6px;margin-right:18px}h2{font-size:19px}a{color:#175bd0}</style>
<header><strong>FewClick 当前C3：分层随机20例</strong><p>四源×五任务各1例，seed=20261021；原质版本，3正2负GT模拟点。开发val/patch级，非WSI盲测。点击图片可查看大图。</p>
<label>数据源 <select id="source"><option value="">全部</option><option>crc</option><option>tiger</option><option>ignite</option><option>ocelot</option></select></label>
<label>任务 <select id="kind"><option value="">全部</option>'''+''.join(f'<option value="{k}">{v}</option>' for k,v in KIND_NAMES.items())+'''</select></label>
<span id="count"></span></header>'''+''.join(cards)+'''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(x=>{let show=(!source.value||x.dataset.source===source.value)&&(!kind.value||x.dataset.kind===kind.value);x.hidden=!show;n+=show});document.querySelector('#count').textContent='显示 '+n+' 例'}const source=document.querySelector('#source'),kind=document.querySelector('#kind');source.onchange=kind.onchange=filter;filter();</script></html>'''
    (output/'index.html').write_text(document)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-run',type=Path,required=True)
    p.add_argument('--checkpoint-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261021);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if ROOT not in a.output.resolve().parents:raise ValueError('output outside artifact root')
    a.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);configure_font()
    try:
        run=json.loads((a.training_run/'run.json').read_text());plans=json.loads((a.training_run/'validation_plans.json').read_text())
        if sha((a.training_run/'validation_plans.json').read_bytes())!=run['validation_plans_sha256']:raise ValueError('validation manifest changed')
        selection=select_plans(plans,a.seed)
        dump(a.output/'selection.json',dict(seed=a.seed,selection_before_scores=True,method='one original-quality task per dataset x kind',all_selected=selection))
        if a.smoke:selection=[s for s in selection if s['plan']['kind']=='local']
        corpus=EpisodeCorpus(Path(run['args']['catalog']),run['args']['seed'])
        if sha((corpus.catalog/'tiles.parquet').read_bytes())!=run['catalog_sha256']:raise ValueError('catalog changed')
        contract=json.loads(a.checkpoint_plan.read_text());info=contract['checkpoints']['C3'];checkpoint=Path(info['path'])
        if checkpoint.parent!=a.training_run or info['step']!=20000 or sha(checkpoint.read_bytes())!=info['sha256']:raise ValueError('locked C3 changed')
        initial,weights=load_model(torch.device('cuda:0'));initial.eval().requires_grad_(False)
        if weights!=run['weights']:raise ValueError('initial model changed')
        current=copy.deepcopy(initial);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['step']!=20000 or saved['mapper']!='pixel':raise ValueError('C3 checkpoint contract changed')
        current.load_state_dict(saved['model'],strict=True);current.eval().requires_grad_(False);del saved
        reference={name:json.loads((a.training_run/folder/'rows.json').read_text()) for name,folder in [('initial','baseline'),('C3','validation_step_20000')]}
        reference={name:{(r['number'],r['quality_condition']):r for r in rows} for name,rows in reference.items()}
        amp=not run['args']['no_amp'];items=[]
        dump(a.output/'run.json',dict(worker_pid=os.getpid(),source_revision=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            source_sha256=sha(Path(__file__).read_bytes()),dependencies={str(Path(inspect.getfile(f))):sha(Path(inspect.getfile(f)).read_bytes()) for f in (load_model,EpisodeCorpus)},
            training_run=str(a.training_run),checkpoint=info,seed=a.seed,smoke=a.smoke,threshold=.5,amp=amp,puma_used=False,test_used=False))
        for number,s in enumerate(selection):
            plan=s['plan'];inputs,supervision=corpus.materialize(plan,'cuda:0');arrays={k:supervision[k].cpu().numpy()[0] for k in ('truth','valid','support_truth','support_valid')}
            for sign,identity,quality in [('support',plan['support_id'],plan['support_quality']),('query',plan['query_id'],plan['query_quality'])]:
                row=corpus.rows[corpus.index[identity]];variant=json.loads(row['variants_json'])[quality]
                arrays[sign+'_rgb']=np.asarray(Image.open(variant['image']).convert('RGB'))
            item=dict(s,initial=None,C3=None)
            with torch.inference_mode():
                for name,model in [('initial',initial),('C3',current)]:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):result=model(**inputs,with_support=False)
                    prob=result['query']['pixel_probability'].float()
                    if not torch.isfinite(prob).all():raise FloatingPointError('nonfinite output')
                    if not inputs['local'] and result['query_prompt_maps'].count_nonzero():raise ValueError('remote point map contamination')
                    counts={k:int(v[0]) for k,v in binary_counts(prob>=.5,supervision['truth'],supervision['valid']).items()}
                    old=reference[name][(plan['number'],plan['quality_condition'])]
                    if any(counts[k]!=old[k] for k in counts):raise ValueError(f'validation count replay mismatch: {name} {plan["number"]}')
                    if any(old[k]!=v for k,v in plan.items()):raise ValueError('reference task identity mismatch')
                    item[name]=old;arrays[name+'_probability']=prob.cpu().numpy()[0]
            folder=a.output/f"case_{s['case']:02d}";folder.mkdir();np.savez_compressed(folder/'arrays.npz',**arrays)
            with np.load(folder/'arrays.npz') as check:
                v=check['valid'];t=check['truth']&v
                for name in ('initial','C3'):
                    pred=(check[name+'_probability']>=.5)&v;actual=(int((pred&t).sum()),int((pred&~t&v).sum()),int((~pred&t).sum()))
                    if actual!=tuple(item[name][k] for k in ('tp','fp','fn')):raise ValueError('saved prediction counts changed')
            dump(folder/'episode.json',item);render_case(folder,item);items.append(item)
            dump(a.output/'status.json',dict(status='rendering',worker_pid=os.getpid(),completed=number+1,total=len(selection),seconds=time.time()-started))
            print('CASE',s['case'],plan['dataset'],plan['kind'],'old',score_title(item['initial']),'C3',score_title(item['C3']),flush=True)
        if not a.smoke:make_overviews(a.output,items)
        write_gallery(a.output,items);dump(a.output/'cases.json',items)
        dump(a.output/'summary.json',dict(cases=len(items),present=sum(r['C3']['positive']>0 for r in items),absent=sum(r['C3']['positive']==0 for r in items),
             all_replay_counts_exact=True,saved_counts_exact=True,selection_before_scores=True,puma_used=False,test_used=False))
        dump(a.output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-started));print('COMPLETE',flush=True)
    except Exception as exc:
        dump(a.output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':main()
