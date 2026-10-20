"""Fixed-FOV CRC RGB downsample/restore control with unchanged cell inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import module_source
from .boundary_data import load_baseline, make_loader, move_batch
from .continue_dense_boundary import DenseContextModel, dense_loss
from .evaluate_dense_transfer import ROOT, DENSE_RUN, IMAGE_KEYS, PROMPT_KEYS, sha256
from .probe_boundary_refinement import metrics, summarize
from .support_query_dense import encode_image, encode_support, predict_query
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v4.phase_6_mask_decoder.src.joint_dataset import collate_joint_pixel_episodes

SCALE_AUDIT=ROOT/'dataset_audit/scale_match_1016_220050/scale_audit_1016_220050.json'
CONDITIONS=('original','degraded_full','degraded_dense_only')


def select_positions(refs, per_class, seed=20261016):
    rng=np.random.default_rng(seed);selected=[]
    for c in sorted({int(r['identity']['target_class']) for r in refs}):
        candidates=[i for i,r in enumerate(refs) if int(r['identity']['target_class'])==c]
        selected.extend(map(int,rng.choice(candidates,min(per_class,len(candidates)),replace=False)))
    return sorted(selected)


def input_rgb(tensor):
    mean=tensor.new_tensor([.485,.456,.406])[None,:,None,None]
    std=tensor.new_tensor([.229,.224,.225])[None,:,None,None]
    return ((tensor*std+mean).clamp(0,1)*255).round().to(torch.uint8)[0].permute(1,2,0).cpu().numpy()


def degrade_rgb(rgb,mpp_x,mpp_y,target_mpp):
    h,w=rgb.shape[:2]
    if min(mpp_x,mpp_y)<=0 or target_mpp<=max(mpp_x,mpp_y):raise ValueError('requires physical downsampling')
    size=(max(1,round(w*mpp_x/target_mpp)),max(1,round(h*mpp_y/target_mpp)))
    low=Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)
    restored=low.resize((w,h),Image.Resampling.BILINEAR)
    return np.array(restored),np.array(low)


def summarize_resolution(rows):
    report={}
    for condition in CONDITIONS:
        report[condition]={}
        for model in ('j5','dense'):
            rr=[r for r in rows if r['condition']==condition and r['model']==model]
            stats=summarize([r|{'prompt_violations':0} for r in rr]);stats.pop('prompt_violations')
            stats['per_class']={str(c):summarize([r|{'prompt_violations':0} for r in rr if r['target_class']==c]) for c in sorted({r['target_class'] for r in rr})}
            for class_stats in stats['per_class'].values():class_stats.pop('prompt_violations')
            original={r['position']:r for r in rows if r['condition']=='original' and r['model']==model}
            stats['paired']={}
            for key in ('dice','bf1_2'):
                evaluable=[r for r in rr if r[key] is not None and original[r['position']][key] is not None]
                groups=sorted({r['spatial_group'] for r in evaluable})
                if not groups:stats['paired'][key]={'evaluable':0};continue
                values=[[r[key]-original[r['position']][key] for r in evaluable if r['spatial_group']==g] for g in groups]
                sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values]);rng=np.random.default_rng(20261016)
                draws=rng.integers(0,len(groups),(2000,len(groups)));boot=sums[draws].sum(1)/counts[draws].sum(1)
                stats['paired'][key]={'degraded_minus_original':float(sums.sum()/counts.sum()),'pairs':len(evaluable),
                    'spatial_blocks':len(groups),'spatial_block_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist()}
            report[condition][model]=stats
    return report


def draw_example(folder,truth,valid,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images=[np.array(Image.open(folder/n)) for n in ('original.png','restored.png')]
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,im,title in zip(axes[:2],images,('Original RGB','Downsampled then restored')):ax.imshow(im);ax.set_title(title)
    for ax,key,title in [(axes[2],'original_dense','Original dense'),(axes[3],'degraded_full_dense','Degraded full dense')]:
        ax.imshow(images[key!='original_dense']);ax.contour(truth,levels=[.5],colors=['lime'],linewidths=.7)
        pred=arrays[key]&valid
        if pred.any() and not pred.all():ax.contour(pred,levels=[.5],colors=['darkorange'],linewidths=.7)
        ax.set_title(title)
    for ax in axes:ax.axis('off')
    fig.suptitle('Same FOV, GT, clicks and cell cache; green GT, orange prediction')
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=140,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if ROOT not in args.output.resolve().parents:raise ValueError('artifact root')
    args.output.mkdir(parents=True,exist_ok=False);started=time.time();torch.set_num_threads(4);device=torch.device('cuda:0')
    source=json.loads((DENSE_RUN/'run.json').read_text());original=json.loads((Path(source['source_cache'])/'run.json').read_text())
    metadata=json.loads((Path(source['checkpoint']).parent/'run_metadata.json').read_text())
    da=argparse.Namespace(**original['args']);da.spatial_data=Path(da.spatial_data);da.routing=Path(da.routing)
    dataset=make_loader(da,metadata,'val').dataset.dataset
    spatial=pd.read_parquet(da.spatial_data/'spatial_manifest.parquet').set_index('patch_id')
    scale=json.loads(SCALE_AUDIT.read_text());scales={r['patch_id']:r for r in scale['crc_patches'] if r['split']=='val'}
    target=float(scale['summary']['ocelot']['native_tissue_mpp_x']['median'])
    positions=select_positions(source['refs']['val'],1 if args.smoke else 8)
    manifest={'source_revision':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        'code_sha256':sha256(module_source('evaluate_crc_resolution')),'scale_audit':str(SCALE_AUDIT),'scale_sha256':sha256(SCALE_AUDIT),
        'target_mpp':target,'refs':[{'position':i,**source['refs']['val'][i]} for i in positions],
        'seed':20261016,'smoke':args.smoke,'test_used':False,'train':False,'cells_policy':'original frozen cache unchanged in all conditions',
        'GT_policy':'original query target/valid unchanged; no GT resampling','worker_pid':os.getpid(),
        'baseline_checkpoint':source['checkpoint'],'dense_checkpoint':str(DENSE_RUN/'dense128_epoch_031.pth'),
        'dense_sha256':sha256(DENSE_RUN/'dense128_epoch_031.pth'),'baseline_sha256':sha256(Path(source['checkpoint']))}
    (args.output/'run.json').write_text(json.dumps(manifest,indent=2));(args.output/'status.json').write_text(json.dumps({'status':'loading','worker_pid':os.getpid()}))
    j5=load_baseline(metadata,Path(source['checkpoint']),device).eval().requires_grad_(False);dense=DenseContextModel().to(device).eval().requires_grad_(False)
    dense.load_state_dict(torch.load(DENSE_RUN/'dense128_epoch_031.pth',map_location=device,weights_only=False)['model'],strict=True)
    rows=[];replay=[];checks={}
    for number,position in enumerate(positions):
        ref=source['refs']['val'][position];identity=ref['identity'];batch=collate_joint_pixel_episodes([dataset[int(identity['episode_index'])]],max_cells=255)
        if str(batch['patch_id'][0])!=identity['patch_id'] or int(batch['target_class'][0])!=identity['target_class']:raise ValueError('episode identity')
        pixel_gt=batch['pixel_gt'];truth=(pixel_gt==identity['target_class']).to(device);valid=(pixel_gt!=255).to(device)
        original_inputs=move_batch({k:batch[k] for k in IMAGE_KEYS},device);prompts=move_batch({k:batch[k] for k in PROMPT_KEYS},device)
        rgb=input_rgb(original_inputs['image']);sc=scales[identity['patch_id']]
        restored,low=degrade_rgb(rgb,sc['input_mpp_x'],sc['input_mpp_y'],target)
        degraded_inputs=original_inputs|{'image':he_to_tensor(restored)[None].to(device)}
        for k in IMAGE_KEYS:
            if k!='image' and not torch.equal(original_inputs[k],degraded_inputs[k]):raise ValueError('cell/active input changed')
        folder=args.output/f'{position:04d}';folder.mkdir();Image.fromarray(rgb).save(folder/'original.png');Image.fromarray(restored).save(folder/'restored.png');Image.fromarray(low).save(folder/'intermediate.png')
        with torch.inference_mode():
            e=encode_image(j5,original_inputs);task=encode_support(j5,e,prompts)
            pred0,features0=predict_query(j5,dense,e,task,prompts,query_id=identity['patch_id'],support_id=identity['patch_id'])
            old=torch.load(ref['file'],map_location=device,weights_only=False)
            if not torch.equal(truth,old['truth']) or not torch.equal(valid,old['valid']):raise ValueError('GT replay')
            error={'position':position,'j5':float((pred0['j5']-old['base']).abs().max()),'dense':float((pred0['dense']-dense(old)['pixel_probability']).abs().max())}
            if max(error['j5'],error['dense'])>2e-6:raise ValueError(f'original replay {error}')
            replay.append(error)
            ed=encode_image(j5,degraded_inputs);td=encode_support(j5,ed,prompts)
            pred1,_=predict_query(j5,dense,ed,td,prompts,query_id=identity['patch_id'],support_id=identity['patch_id'])
            fixed=features0|{k:ed[k] for k in ('deep','shallow','semantic')}
            predictions={'original':pred0,'degraded_full':pred1,'degraded_dense_only':{'j5':pred0['j5'],'dense':dense(fixed)['pixel_probability']}}
            if args.smoke:
                loss=dense_loss(dense(fixed),{'truth':truth,'valid':valid})
                if not torch.isfinite(loss):raise ValueError('real degraded loss')
                checks['finite_degraded_loss']=True
        arrays={'truth':truth.cpu().numpy()[0],'valid':valid.cpu().numpy()[0]};current=[]
        for condition,pp in predictions.items():
            for model,probability in pp.items():
                if not torch.isfinite(probability).all():raise ValueError('nonfinite prediction')
                row={'position':position,'patch_id':identity['patch_id'],'target_class':identity['target_class'],'condition':condition,'model':model,
                     'spatial_group':str(spatial.loc[identity['patch_id'],'spatial_group_id']),**metrics(probability,truth,valid)}
                current.append(row);arrays[condition+'_'+model]=(probability>=.5).cpu().numpy()[0]
        np.savez_compressed(folder/'predictions.npz',**arrays)
        with np.load(folder/'predictions.npz') as a:
            v,t=a['valid'],a['truth']&a['valid']
            for row in current:
                m=a[row['condition']+'_'+row['model']]&v
                if (int((m&t).sum()),int((m&~t&v).sum()),int((~m&t).sum()))!=(row['tp'],row['fp'],row['fn']):raise ValueError('saved mask count')
            if not np.array_equal(a['original_j5'],a['degraded_dense_only_j5']):raise ValueError('fixed J5 changed')
        rows.extend(current)
        details={'identity':identity,'original_mpp':[sc['input_mpp_x'],sc['input_mpp_y']],'target_mpp':target,'intermediate_size':list(low.shape[:2][::-1]),
            'effective_intermediate_mpp':[512*sc['input_mpp_x']/low.shape[1],512*sc['input_mpp_y']/low.shape[0]],
            'original_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'degraded_rgb_sha256':hashlib.sha256(restored.tobytes()).hexdigest(),
            'replay':error,'prompts':{k:v.cpu().tolist() for k,v in prompts.items()},'cells_unchanged':True}
        (folder/'details.json').write_text(json.dumps(details,indent=2))
        # A class-fixed example is saved for every selected query, never by score.
        draw_example(folder,arrays['truth'],arrays['valid'],arrays)
        with (args.output/'metrics.jsonl').open('a') as f:
            for row in current:f.write(json.dumps(row)+'\n')
        with (args.output/'completed.jsonl').open('a') as f:f.write(json.dumps(details)+'\n')
        (args.output/'status.json').write_text(json.dumps({'status':'evaluating','completed':number+1,'cases':len(positions),'worker_pid':os.getpid(),'elapsed_seconds':time.time()-started}))
        print('CASE',number+1,'/',len(positions),'seconds',round(time.time()-started,1),flush=True)
    report={'summary':summarize_resolution(rows),'cases':len(positions),'target_mpp':target,'smoke':args.smoke,'checks':checks,
        'max_original_j5_replay_error':max(r['j5'] for r in replay),'max_original_dense_replay_error':max(r['dense'] for r in replay),
        'saved_mask_metrics_verified':True,'cells_unchanged':True,'elapsed_seconds':time.time()-started}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2));(args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
