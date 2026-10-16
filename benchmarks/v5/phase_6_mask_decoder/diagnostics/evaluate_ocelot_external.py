"""Six-organ, within-ROI separated support/query frozen-model input pilot."""
import argparse
from collections import Counter
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
from . import module_source
from .boundary_data import load_baseline
from .continue_dense_boundary import DenseContextModel, dense_loss
from .evaluate_dense_transfer import ROOT, DENSE_RUN, sha256
from .external_cell_inputs import WEIGHTS, load_cell_models, prepare_cells
from .probe_boundary_refinement import metrics, summarize
from .support_query_dense import encode_image, encode_support, predict_query
from .evaluate_prompt_dependence import zero_task
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor

DATA = Path('/nfs-medical3/zyh/OCELOT_1016_122128/extracted/ocelot2023_v1.0.1')


def points(mask, count):
    distance=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    xy=[];yy,xx=np.indices(mask.shape)
    for _ in range(count):
        y,x=np.unravel_index(distance.argmax(),distance.shape)
        if distance[y,x]<=0:raise ValueError('insufficient separated support points')
        xy.append([(x+.5)/mask.shape[1],(y+.5)/mask.shape[0]])
        distance[(xx-x)**2+(yy-y)**2<16**2]=0
    return xy


def support_prompts(labels, target):
    other=1 if target==2 else 2
    # Small prompt contract: 3 positive, 2 negative; same budget after switching.
    return {'positive_xy':[points(labels==target,3)],'negative_xy':[points(labels==other,2)],
        'positive_mask':[[True]*3],'negative_mask':[[True]*2],'prompt_size_id':[1]}


def crop_boxes(width,height):
    if width<512 or height<512:return []
    corners=[(0,0),(width-512,height-512),(width-512,0),(0,height-512)]
    result=[]
    for a,b in [(corners[0],corners[1]),(corners[1],corners[0]),(corners[2],corners[3]),(corners[3],corners[2])]:
        gap=np.hypot(max(0,abs(a[0]-b[0])-512),max(0,abs(a[1]-b[1])-512))*.5
        if gap>=128:result.append(([*a,a[0]+512,a[1]+512],[*b,b[0]+512,b[1]+512],float(gap)))
    return result


def prepare(output):
    metadata=json.loads((DATA/'metadata.json').read_text());selected=[];skipped=[];counts=Counter();seen=set()
    for identity,meta in sorted(metadata['sample_pairs'].items()):
        if meta['subset']!='val':continue
        organ=meta['organ'];slide=meta['slide_name']
        if counts[organ]>=4 or (organ,slide) in seen:continue
        image_path=DATA/'images/val/tissue'/f'{identity}.jpg'
        gt_path=DATA/'annotations/val/tissue'/f'{identity}.png'
        with Image.open(image_path) as im:
            original_size=im.size
            size=tuple(round(original_size[i]*meta['tissue'][f'resized_mpp_{axis}']/.5) for i,axis in enumerate(('x','y')))
            rgb=np.array(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))
        with Image.open(gt_path) as im:
            if im.size!=original_size:raise ValueError('source GT/image extent')
            gt=np.array(im.resize(size,Image.Resampling.NEAREST))
        if set(np.unique(gt))-{1,2,255}:raise ValueError('OCELOT labels')
        found=None
        for sb,qb,gap in crop_boxes(*size):
            sl=gt[sb[1]:sb[3],sb[0]:sb[2]]
            if (sl!=255).mean()<.9 or min(int((sl==1).sum()),int((sl==2).sum()))<64:continue
            try:prompts={str(c):support_prompts(sl,c) for c in (1,2)}
            except ValueError:continue
            found=(sb,qb,gap,prompts);break
        if found is None:
            skipped.append({'id':identity,'organ':organ,'reason':'no strict separated, mixed-class support pair'});continue
        sb,qb,gap,prompts=found;folder=output/'patches'/identity;folder.mkdir(parents=True)
        paths={}
        for role,box in [('support',sb),('query',qb)]:
            Image.fromarray(rgb[box[1]:box[3],box[0]:box[2]]).save(folder/f'{role}.png')
            Image.fromarray(gt[box[1]:box[3],box[0]:box[2]]).save(folder/f'{role}_gt.png')
            paths[role]={'image':str(folder/f'{role}.png'),'gt':str(folder/f'{role}_gt.png'),'box_resampled':box,
                'box_source_roi':[box[0]*original_size[0]/size[0],box[1]*original_size[1]/size[1],box[2]*original_size[0]/size[0],box[3]*original_size[1]/size[1]]}
        selected.append({'id':identity,'organ':organ,'slide':slide,'source_subset':'val','source_image':str(image_path),
            'source_gt':str(gt_path),'metadata':meta,'original_size':original_size,'resampled_size':size,
            'nominal_mpp':.5,'effective_mpp':[meta['tissue'][f'resized_mpp_{axis}']*original_size[i]/size[i] for i,axis in enumerate(('x','y'))],
            'gap_um':gap,'prompts':prompts,**paths})
        counts[organ]+=1;seen.add((organ,slide))
    report={'dataset':'OCELOT 1.0.1','metadata_sha256':sha256(DATA/'metadata.json'),'cases':selected,'skipped':skipped,
        'organ_counts':dict(counts),'max_cases_per_organ':4,'selection':'val only, one ROI per slide, sorted ID, before predictions',
        'target_label':2,'ignore':255,'prompt_budget':'3 positive + 2 negative, small',
        'scope':'external-organ within-ROI separated patches; unknown nuclei class, interpolated detection image; not whole WSI'}
    (output/'selection.json').write_text(json.dumps(report,indent=2))
    print('PREPARED',dict(counts),'skipped',len(skipped),flush=True)


def report_rows(rows):
    output={}
    for organ in ['all',*sorted({r['organ'] for r in rows})]:
        output[organ]={}
        for model in ('j5','dense'):
            output[organ][model]={}
            for condition in ('cancer_support','noncancer_support','zero_task'):
                rr=[r for r in rows if r['model']==model and r['condition']==condition and (organ=='all' or r['organ']==organ)]
                present=[r for r in rr if r['target_present']];absent=[r for r in rr if not r['target_present']]
                stats={'present':summarize([r|{'prompt_violations':0} for r in present]) if present else {'episodes':0},
                    'absent':{'episodes':len(absent),**{k:float(np.mean([r[k] for r in absent])) if absent else None for k in ('false_positive_area','area_activation_1pct','any_activation')}}}
                stats['present'].pop('prompt_violations',None);output[organ][model][condition]=stats
    return output


def draw_case(folder,case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(20,4))
    support=np.array(Image.open(case['support']['image']));query=np.array(Image.open(case['query']['image']))
    axes[0].imshow(support)
    for sign,color,marker in [('positive','deepskyblue','+'),('negative','magenta','x')]:
        xy=np.array(case['prompts']['2'][sign+'_xy'][0])*512
        axes[0].scatter(xy[:,0],xy[:,1],c=color,marker=marker,s=70)
    axes[0].set_title('Support: 3 positive + 2 negative')
    with np.load(folder/'predictions.npz') as data:
        for ax,key,title in [(axes[1],'truth','Query cancer GT'),(axes[2],'cancer_support_j5','J5: cancer support'),
                             (axes[3],'cancer_support_dense','Dense: cancer support'),(axes[4],'noncancer_support_dense','Dense: noncancer support')]:
            ax.imshow(query);ax.contour(data['truth'],levels=[.5],colors=['lime'],linewidths=.8)
            if key!='truth':ax.contour(data[key],levels=[.5],colors=['darkorange'],linewidths=.8)
            else:ax.imshow(np.ma.masked_where(~data['truth'],data['truth']),cmap='Greens',vmin=0,vmax=1,alpha=.3)
            ax.set_title(title)
    for ax in axes:ax.axis('off')
    fig.suptitle(f"{case['organ']} | ROI {case['id']} | same slide, disjoint crops | green: cancer GT; orange: prediction")
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--manifest',type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if ROOT not in args.output.resolve().parents:raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.prepare:prepare(args.output);return
    if args.manifest is None:raise ValueError('explicit manifest required')
    selection=json.loads(args.manifest.read_text());cases=selection['cases'][:1] if args.smoke else selection['cases']
    if not cases:raise ValueError('empty selection')
    torch.set_num_threads(4);torch.manual_seed(20261016);device=torch.device('cuda:0');started=time.time();smoke_checks={}
    (args.output/'status.json').write_text(json.dumps({'status':'loading','worker_pid':os.getpid()}))
    run=json.loads((DENSE_RUN/'run.json').read_text());metadata=json.loads((Path(run['checkpoint']).parent/'run_metadata.json').read_text())
    j5=load_baseline(metadata,Path(run['checkpoint']),device).eval().requires_grad_(False)
    dense=DenseContextModel().to(device).eval().requires_grad_(False)
    checkpoint=DENSE_RUN/'dense128_epoch_031.pth';dense.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model'],strict=True)
    detector,ctp,xcell,audit=load_cell_models(device)
    manifest={'selection':str(args.manifest),'selection_sha256':sha256(args.manifest),'cases':cases,'scope':selection['scope'],
        'weights':{k:{'path':str(v),'sha256':sha256(v)} for k,v in WEIGHTS.items()},'dense_sha256':sha256(checkpoint),
        'j5_sha256':sha256(Path(run['checkpoint'])),'cell_model_audit':audit,
        'code_revision':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        'source_sha256':{n:sha256(module_source(n)) for n in ('evaluate_ocelot_external','external_cell_inputs','support_query_dense')},
        'train':False,'test_used':False,'smoke':args.smoke,'worker_pid':os.getpid(),'threshold':.5,
        'absent_policy':'No cancer in labeled valid query area; unknown pixels are excluded, not negative',
        'input_hashes':{c['id']:{role:{k:sha256(Path(c[role][k])) for k in ('image','gt')} for role in ('support','query')} for c in cases}}
    (args.output/'run.json').write_text(json.dumps(manifest,indent=2));rows=[]
    for number,case in enumerate(cases):
        folder=args.output/case['id'];folder.mkdir();encoded={};labels={}
        for role in ('support','query'):
            image=np.array(Image.open(case[role]['image']).convert('RGB'));gt=np.array(Image.open(case[role]['gt']))
            if image.shape!=(512,512,3) or gt.shape!=(512,512):raise ValueError('prepared extent')
            cell_inputs,raw=prepare_cells(image,detector,ctp,xcell,device)
            np.savez_compressed(folder/f'{role}_cells.npz',**raw)
            with torch.inference_mode():
                model_inputs={'image':he_to_tensor(image)[None].to(device),**cell_inputs}
                encoded[role]=encode_image(j5,model_inputs)
                if args.smoke and role=='query':
                    contaminated=model_inputs|{'pixel_gt':torch.zeros(1,512,512,device=device),'target_class':torch.tensor([999],device=device)}
                    repeated=encode_image(j5,contaminated)
                    if any(not torch.equal(v,repeated[k]) for k,v in encoded[role].items()):raise ValueError('query GT independence')
                    smoke_checks['query_gt_independent']=True
            torch.save({k:v.cpu() for k,v in encoded[role].items()},folder/f'{role}_encoded.pth')
            labels[role]=gt
            print('ENCODED',number+1,role,'cells',raw['total'],'seconds',round(time.time()-started,1),flush=True)
        prompts={c:{k:torch.tensor(v,device=device) for k,v in case['prompts'][str(c)].items()} for c in (1,2)}
        task=encode_support(j5,encoded['support'],prompts[2]);tasks={'cancer_support':task,
            'noncancer_support':encode_support(j5,encoded['support'],prompts[1]),'zero_task':zero_task(task)}
        truth=torch.from_numpy(labels['query']==2)[None].to(device);valid=torch.from_numpy(labels['query']!=255)[None].to(device)
        if not valid.any():raise ValueError('no scored query area')
        arrays={'truth':truth.cpu().numpy()[0],'valid':valid.cpu().numpy()[0]}
        with torch.inference_mode():
            for condition,task in tasks.items():
                pred,features=predict_query(j5,dense,encoded['query'],task,prompts[2],query_id='query',support_id='support')
                if features['prompt_maps'].count_nonzero():raise ValueError('remote click leakage')
                for name,probability in pred.items():
                    if not torch.isfinite(probability).all():raise ValueError('non-finite prediction')
                    if args.smoke and condition=='cancer_support' and name=='dense':
                        loss=dense_loss(dense(features),{'truth':truth,'valid':valid})
                        if not torch.isfinite(loss):raise ValueError('real smoke loss')
                        smoke_checks['real_dense_loss']=float(loss)
                    row={'id':case['id'],'organ':case['organ'],'slide':case['slide'],'model':name,'condition':condition,
                         'target_present':bool(truth.any()),'valid_fraction':float(valid.float().mean()),'target_pixels':int(truth.sum())}
                    if truth.any():row.update(metrics(probability,truth,valid))
                    else:
                        area=float(((probability>=.5)&valid).sum()/valid.sum().clamp_min(1))
                        row.update(false_positive_area=area,any_activation=area>0,area_activation_1pct=area>=.01)
                    rows.append(row);arrays[condition+'_'+name]=(probability>=.5).cpu().numpy()[0]
        np.savez_compressed(folder/'predictions.npz',**arrays)
        with np.load(folder/'predictions.npz') as saved:
            for row in rows[-6:]:
                pred=saved[row['condition']+'_'+row['model']]&saved['valid'];t=saved['truth']&saved['valid']
                if row['target_present']:
                    tp,fp,fn=int((pred&t).sum()),int((pred&~t).sum()),int((~pred&t).sum())
                    if (tp,fp,fn)!=(row['tp'],row['fp'],row['fn']):raise ValueError('mask metric mismatch')
                elif abs(float(pred.sum()/max(int(saved['valid'].sum()),1))-row['false_positive_area'])>1e-7:raise ValueError('absent mask metric mismatch')
        draw_case(folder,case)
        with (args.output/'metrics.jsonl').open('a') as f:
            for row in rows[-6:]:f.write(json.dumps(row)+'\n')
        with (args.output/'completed.jsonl').open('a') as f:f.write(json.dumps({'number':number,'id':case['id']})+'\n')
        (args.output/'status.json').write_text(json.dumps({'status':'evaluating','worker_pid':os.getpid(),'completed':number+1,'cases':len(cases),'elapsed_seconds':time.time()-started}))
        print('CASE',number+1,'/',len(cases),'seconds',round(time.time()-started,1),flush=True)
    report={'summary':report_rows(rows),'cases':len(cases),'organ_counts':dict(Counter(c['organ'] for c in cases)),
        'scope':manifest['scope'],'smoke':args.smoke,'smoke_checks':smoke_checks,'saved_mask_metrics_verified':True,'elapsed_seconds':time.time()-started}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
