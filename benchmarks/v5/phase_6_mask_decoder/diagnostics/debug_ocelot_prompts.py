"""Frozen factorial controls for support-pixel locations versus region mapping."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import pyvips
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from . import module_source
from .boundary_data import load_baseline
from .continue_dense_boundary import DenseContextModel
from .evaluate_dense_transfer import ROOT, DENSE_RUN, sha256
from .evaluate_ocelot_external import report_rows
from .probe_boundary_refinement import metrics
from .support_query_dense import encode_support, predict_query

SOURCE=ROOT/'support_query/ocelot_external_1016_201057/formal_1016_201057'
ARMS=('original','negative_spread','both_spread','pixel_mapping','spread_pixel_mapping','tissue_negative')


def spread_points(mask,count):
    distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    eligible=distance>=4;yy,xx=np.indices(mask.shape);selected=[]
    for i in range(count):
        if not eligible.any():raise ValueError('not enough source interior points >=64px apart')
        if not selected:score=distance.copy()
        else:score=np.minimum.reduce([(xx-x)**2+(yy-y)**2 for y,x in selected]).astype(float)
        score[~eligible]=-1;y,x=np.unravel_index(score.argmax(),score.shape);selected.append((y,x))
        eligible&=((xx-x)**2+(yy-y)**2>=64**2)
    return [[(float(x)+.5)/mask.shape[1],(float(y)+.5)/mask.shape[0]] for y,x in selected]


def make_variants(original,labels,image):
    result={'original':original,'pixel_mapping':original};skipped={}
    def change(positive=False,tissue=False):
        out={k:json.loads(json.dumps(v)) for k,v in original.items()}
        negative=(labels==1)
        if tissue:
            rgb=image.astype(np.float32)/255;mx=rgb.max(2);mn=rgb.min(2)
            negative&=((mx-mn)/np.maximum(mx,1e-6)>=.08)&(rgb.mean(2)<.95)
        out['negative_xy']=[spread_points(negative,2)]
        if positive:out['positive_xy']=[spread_points(labels==2,3)]
        return out
    for arm,kwargs in [('negative_spread',{}),('both_spread',{'positive':True}),('tissue_negative',{'tissue':True})]:
        try:result[arm]=change(**kwargs)
        except ValueError as error:skipped[arm]=str(error)
    if 'both_spread' in result:result['spread_pixel_mapping']=result['both_spread']
    else:skipped['spread_pixel_mapping']=skipped['both_spread']
    return result,skipped


def pixel_task(model,encoded,prompts):
    values={'prompt_size_id':prompts['prompt_size_id']};indices={}
    # Match the full-resolution assignment used for query pixel projection.
    full=F.interpolate(encoded['assignment_low'],size=(512,512),mode='bilinear',align_corners=False)
    for sign in ('positive','negative'):
        xy=prompts[sign+'_xy'];valid=prompts[sign+'_mask']
        mass=F.grid_sample(full,(xy*2-1).unsqueeze(1),mode='bilinear',align_corners=False,padding_mode='border').squeeze(2).transpose(1,2)
        ids=mass.argmax(-1);tokens=encoded['tokens'].gather(1,ids[...,None].expand(-1,-1,encoded['tokens'].shape[-1]))
        values[sign+'_tokens']=tokens*valid[...,None]
        values[sign+'_xy']=xy;values[sign+'_mask']=valid;indices[sign]=ids.masked_fill(~valid,-1).cpu().tolist()
    return model.decoder.prompt_model.encode_prompt_task(values),indices


def summarize_debug(rows):
    # Reuse present/absent reporting after remapping the arbitrary arm labels.
    report={};baseline={(r['id'],r['model']):r for r in rows if r['arm']=='original'}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        normalized=[r|{'condition':'cancer_support'} for r in subset]
        report[arm]=report_rows(normalized) if subset else {}
        for model in ('j5','dense'):
            pp=[r for r in subset if r['model']==model and r['target_present']]
            aa=[r for r in subset if r['model']==model and not r['target_present']]
            if 'all' in report[arm]:
                stats=report[arm]['all'][model]['cancer_support']
                stats['present']['original_dice_on_same_subset']=float(np.mean([baseline[(r['id'],model)]['dice'] for r in pp])) if pp else None
                stats['present']['paired_dice_delta']=float(np.mean([r['dice']-baseline[(r['id'],model)]['dice'] for r in pp])) if pp else None
                stats['absent']['original_fpa_on_same_subset']=float(np.mean([baseline[(r['id'],model)]['false_positive_area'] for r in aa])) if aa else None
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if ROOT not in args.output.resolve().parents:raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.time();device=torch.device('cuda:0')
    source=json.loads((SOURCE/'run.json').read_text());cases=source['cases'][:1] if args.smoke else source['cases']
    plan=[]
    for case in cases:
        labels=np.array(Image.open(case['support']['gt']));image=np.array(Image.open(case['support']['image']).convert('RGB'))
        variants,skipped=make_variants(case['prompts']['2'],labels,image)
        plan.append({'case':case,'prompts':variants,'skipped':skipped})
    # Freeze the selection before any prediction/query GT is read.
    manifest={'source':str(SOURCE),'source_sha256':sha256(SOURCE/'run.json'),'plans':plan,'smoke':args.smoke,'train':False,'test_used':False,
        'code_revision':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'code_sha256':sha256(module_source('debug_ocelot_prompts')),
        'policy':'support GT/RGB only for positions; same 3 positive/2 negative small budget; query GT only scoring','worker_pid':os.getpid()}
    (args.output/'run.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'loading','worker_pid':os.getpid()}))
    run=json.loads((DENSE_RUN/'run.json').read_text());meta=json.loads((Path(run['checkpoint']).parent/'run_metadata.json').read_text())
    j5=load_baseline(meta,Path(run['checkpoint']),device).eval().requires_grad_(False);dense=DenseContextModel().to(device).eval().requires_grad_(False)
    checkpoint=DENSE_RUN/'dense128_epoch_031.pth';dense.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'],strict=True)
    rows=[]
    for n,entry in enumerate(plan):
        c=entry['case'];identity=c['id'];folder=args.output/identity;folder.mkdir()
        se=torch.load(SOURCE/identity/'support_encoded.pth',map_location=device,weights_only=False)
        qe=torch.load(SOURCE/identity/'query_encoded.pth',map_location=device,weights_only=False)
        with np.load(SOURCE/identity/'predictions.npz') as data:saved={k:data[k].copy() for k in data.files}
        truth=torch.from_numpy(saved['truth'])[None].to(device);valid=torch.from_numpy(saved['valid'])[None].to(device)
        arrays={'truth':saved['truth'],'valid':saved['valid']};details={}
        with torch.inference_mode():
            for arm,prompt in entry['prompts'].items():
                pp={k:torch.tensor(v,device=device,dtype=torch.float32 if k.endswith('_xy') else None) for k,v in prompt.items()}
                if arm in ('pixel_mapping','spread_pixel_mapping'):task,idx=pixel_task(j5,se,pp)
                else:task=encode_support(j5,se,pp);idx=None
                pred,features=predict_query(j5,dense,qe,task,pp,query_id='query',support_id='support')
                if features['prompt_maps'].count_nonzero():raise ValueError('source points on query')
                details[arm]={'prompts':prompt,'pixel_mapping_indices':idx}
                for model,probability in pred.items():
                    if not torch.isfinite(probability).all():raise ValueError('non-finite')
                    mask=(probability>=.5).cpu().numpy()[0];arrays[arm+'_'+model]=mask
                    if arm=='original' and not np.array_equal(mask,saved['cancer_support_'+model]):raise ValueError('original binary replay mismatch')
                    row={'id':identity,'organ':c['organ'],'slide':c['slide'],'arm':arm,'model':model,'target_present':bool(truth.any())}
                    if truth.any():row.update(metrics(probability,truth,valid))
                    else:
                        area=float(((probability>=.5)&valid).sum()/valid.sum().clamp_min(1));row.update(false_positive_area=area,any_activation=area>0,area_activation_1pct=area>=.01)
                    rows.append(row)
        np.savez_compressed(folder/'predictions.npz',**arrays);(folder/'prompts.json').write_text(json.dumps(details,indent=2))
        current=[r for r in rows if r['id']==identity]
        with np.load(folder/'predictions.npz') as data:
            v,t=data['valid'],data['truth']&data['valid']
            for row in current:
                m=data[row['arm']+'_'+row['model']]&v
                if row['target_present']:
                    if (int((m&t).sum()),int((m&~t&v).sum()),int((~m&t).sum()))!=(row['tp'],row['fp'],row['fn']):raise ValueError('saved counts')
                elif abs(float(m.sum()/max(int(v.sum()),1))-row['false_positive_area'])>1e-7:raise ValueError('saved absent area')
        with (args.output/'metrics.jsonl').open('a') as f:
            for row in current:f.write(json.dumps(row)+'\n')
        with (args.output/'completed.jsonl').open('a') as f:f.write(json.dumps({'id':identity,'arms':list(entry['prompts'])})+'\n')
        (args.output/'status.json').write_text(json.dumps({'status':'evaluating','completed':n+1,'cases':len(plan),'worker_pid':os.getpid(),'elapsed_seconds':time.time()-started}))
        print('CASE',n+1,'/',len(plan),'seconds',round(time.time()-started,1),flush=True)
    report={'summary':summarize_debug(rows),'cases':len(plan),'original_replay_exact':True,'saved_mask_metrics_verified':True,'smoke':args.smoke,'elapsed_seconds':time.time()-started}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));(args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
