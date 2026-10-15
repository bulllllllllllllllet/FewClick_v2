"""Cache and adapt official pretrained SegNext on fixed FewClick episodes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import module_source
from .segnext_adapter import SegNextAdapter, official_points, rgb_from_normalized
from .continue_dense_boundary import dense_loss
from .train_dense_boundary import evaluate, points
from .probe_boundary_refinement import metrics, summarize, prompt_violations

ARTIFACT=Path('/nfs-medical3/zyh/fewclick_v2')
INPUT=ARTIFACT/'boundary_refinement/next_stage_1014_200033/finetune_1014_200033/run.json'
DENSE=ARTIFACT/'boundary_refinement/pixel_prompt_1015_164227/pilot_1015_164227/baseline_rows.json'
KEYS=['base','truth','valid','positive_xy','negative_xy','positive_mask','negative_mask']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def backbone_sha(model):
    digest=hashlib.sha256()
    for name,tensor in model.model.backbone.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    files=[module_source(m) for m in ['run_segnext','segnext_adapter','continue_dense_boundary','train_dense_boundary']]
    files+=sorted((module_source('segnext_adapter').parent/'segnext_vendor').glob('*.py'))
    return {str(p):sha(p) for p in files}


def validate_refs(refs, smoke):
    ids={s:[r['identity']['patch_id'] for r in rows] for s,rows in refs.items()}
    if set(ids['train']) & set(ids['val']) or any(len(x)!=len(set(x)) for x in ids.values()):
        raise RuntimeError('duplicate patch/split identity')
    if not smoke and (len(refs['train']),len(refs['val']))!=(3000,1024):
        raise RuntimeError('formal data budget')


def row_result(probability,sample,identity):
    row={**identity,**metrics(probability,sample['truth'],sample['valid']),
         'prompt_violations':prompt_violations(probability,sample)}
    row['credible_prompts'],row['credible_violations']=points(probability,sample)
    return row


def prepare(args):
    if args.checkpoint is None:raise ValueError('prepare requires checkpoint')
    run=json.loads(INPUT.read_text());refs=copy.deepcopy(run['refs'])
    if args.smoke:refs={s:rows[:(4 if s=='train' else 2)] for s,rows in refs.items()}
    validate_refs(refs,args.smoke)
    (args.output/'features').mkdir()
    model=SegNextAdapter.from_checkpoint(args.checkpoint).cuda().eval()
    original=None
    if args.smoke:
        if args.upstream is None:raise ValueError('smoke requires upstream parity source')
        sys.path.insert(0,str(args.upstream/'segnext'))
        from isegm.model.is_plainvit_model import PlainVitModel
        original=PlainVitModel(**{k:v['value'] for k,v in model.config['params'].items()}).cuda().eval()
        original.load_state_dict(model.model.state_dict(),strict=True)
    metadata={'args':{**run['args'],'mode':'segnext','smoke':args.smoke,'train_samples':len(refs['train']),'val_samples':len(refs['val'])},
              'refs':refs,'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint),
              'source_run':str(INPUT),'source_sha256':source_hashes(),
              'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'worker_pid':os.getpid(),'cache_dtype':'float32','encoder_shape':[1,768,64,64],
              'image_contract':'same 512 field; inverse ImageNet normalization; official resize to 1024; output 512',
              'prompt_contract':'same cached points; xy normalized -> floor-scored row/col; zero previous mask; no TTA or automatic clicks',
              'attention':'unmodified upstream FP32 attention; no SDPA substitution',
              'test_used':False,'parity':[]}
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2))
    rows=[];regression=[];cache_bytes=0;started=time.time()
    for split in ['train','val']:
        for index,ref in enumerate(refs[split]):
            (args.output/'status.json').write_text(json.dumps({'status':'caching','split':split,'completed':index,'total':len(refs[split]),'worker_pid':os.getpid()}))
            source_sample=torch.load(ref['file'],map_location='cpu',weights_only=False)
            sample={k:source_sample[k].cuda() for k in KEYS+['image']}
            with torch.no_grad():
                feature=model.encode(sample)
                if feature.shape!=(1,768,64,64) or not torch.isfinite(feature).all():raise ValueError('encoder feature contract')
                sample['segnext_features']=feature
                if args.smoke and index==0:
                    old_points=sample['positive_xy'].clone()
                    old_feature=original.get_image_feats(rgb_from_normalized(sample['image']),keep_shape=True)
                    prompt={'points':official_points(sample),'prev_mask':sample['base'].new_zeros(1,1,512,512)}
                    prompt_feature=original.get_prompt_feats((512,512),prompt)
                    expected=original((512,512),old_feature,prompt_feature)['instances'][:,0].sigmoid()
                    actual=model(sample)['pixel_probability']
                    feature_error=float((feature-old_feature).abs().max());prob_error=float((expected-actual).abs().max())
                    if feature_error>2e-6 or prob_error>2e-6:raise RuntimeError('upstream parity mismatch')
                    if not torch.equal(old_points,sample['positive_xy']):raise RuntimeError('mutated prompt')
                    altered={**sample,'truth':~sample['truth'],'valid':~sample['valid'],'target_class':torch.tensor([11],device='cuda'),'base':torch.zeros_like(sample['base'])}
                    if not torch.equal(actual,model(altered)['pixel_probability']):raise RuntimeError('GT/base-dependent inference')
                    metadata['parity'].append({'split':split,'feature_max_error':feature_error,'probability_max_error':prob_error})
                if split=='val':
                    probability=model(sample)['pixel_probability']
                    if not torch.isfinite(probability).all():raise FloatingPointError('pretrained probability')
                    rows.append(row_result(probability,sample,ref['identity']))
                    if ref.get('regression'):regression.append((ref['identity']['patch_id'],probability.cpu().numpy()[0]))
            cache_sample={k:sample[k].cpu() for k in KEYS}
            cache_sample['segnext_features']=feature.cpu()
            destination=args.output/'features'/f'{split}_{index:05d}.pth'
            torch.save(cache_sample,destination);cache_bytes+=destination.stat().st_size
            ref['rgb_source_file']=ref['file'];ref['file']=str(destination)
            with (args.output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps({'split':split,'completed':index+1,'total':len(refs[split]),'seconds':time.time()-started})+'\n')
            if (index+1)%100==0:print('CACHE',split,index+1,'/',len(refs[split]),'seconds',round(time.time()-started,1),flush=True)
    (args.output/'pretrained_rows.json').write_text(json.dumps(rows))
    (args.output/'pretrained_summary.json').write_text(json.dumps(summarize(rows),indent=2))
    if regression:np.savez_compressed(args.output/'pretrained_rows.npz',patch_id=[k for k,_ in regression],probability=np.stack([v for _,v in regression]))
    metadata.update(cache_bytes=cache_bytes,seconds=time.time()-started)
    (args.output/'run.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid(),'cache_bytes':cache_bytes}))
    print('PREPARE_COMPLETE',json.dumps(summarize(rows)),flush=True)


def train(args):
    if args.cache is None or json.loads((args.cache/'status.json').read_text())['status']!='complete':raise RuntimeError('cache incomplete')
    cache=json.loads((args.cache/'run.json').read_text())
    if bool(cache['args']['smoke'])!=args.smoke:raise ValueError('smoke/cache mismatch')
    refs=cache['refs'];validate_refs(refs,args.smoke)
    model=SegNextAdapter.from_checkpoint(cache['checkpoint']).cuda()
    if sha(cache['checkpoint'])!=cache['checkpoint_sha256']:raise RuntimeError('checkpoint digest mismatch')
    run={'args':{**cache['args'],'epochs':args.epochs,'seed':20261014},'refs':refs,
         'cache':str(args.cache),'checkpoint':cache['checkpoint'],'checkpoint_sha256':cache['checkpoint_sha256'],
         'source_sha256':source_hashes(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'worker_pid':os.getpid(),'test_used':False,'baseline_label':'Frozen matched96k dense128',
         'report_fixed_final_epoch':True,'comparison_pairs':[],
         'additional_reference_rows':{'segnext_pretrained':'pretrained_rows.json'},
         'optimizer':{'lr':args.lr,'weight_decay':1e-4,'clip_norm':5,'frozen':'image backbone','trainable':'official prompt embedding, fusion, FPN and head'},
         'budget_note':'same 3k images, fixed prompts and 24k updates; external pretraining/parameters/FLOPs differ',
         'scope':'developmental spatial validation; different pretraining exposure; no test'}
    reference={r['patch_id']:r for r in json.loads(DENSE.read_text())}
    baseline_rows=[reference[r['identity']['patch_id']] for r in refs['val']]
    for row,ref in zip(baseline_rows,refs['val']):
        if row['target_class']!=ref['identity']['target_class'] or row['episode_index']!=ref['identity']['episode_index']:raise RuntimeError('dense reference identity')
    baseline=summarize(baseline_rows)
    baseline['credible_prompts']=sum(r['credible_prompts'] for r in baseline_rows)
    baseline['credible_violations']=sum(r['credible_violations'] for r in baseline_rows)
    (args.output/'baseline_rows.json').write_text(json.dumps(baseline_rows))
    pretrained=json.loads((args.cache/'pretrained_rows.json').read_text())
    (args.output/'pretrained_rows.json').write_text(json.dumps(pretrained))
    for label,path in [('native_final',ARTIFACT/'boundary_refinement/multiscale_1015_133617/pilot_1015_133617/native_epoch_007_rows.json')]:
        lookup={r['patch_id']:r for r in json.loads(path.read_text())}
        filename=label+'_rows.json';(args.output/filename).write_text(json.dumps([lookup[r['patch_id']] for r in baseline_rows]))
        run['additional_reference_rows'][label]=filename
    (args.output/'run.json').write_text(json.dumps(run,indent=2))
    model.eval();initial,initial_rows=evaluate(model,refs['val'],'cuda',args.output/'initial_rows.json')
    for actual,expected in zip(initial_rows,pretrained):
        for metric in ['dice','bf1_2']:
            if abs(actual[metric]-expected[metric])>1e-7:raise RuntimeError('FP32 cache/pretrained replay mismatch')
    frozen_before=backbone_sha(model)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=1e-4)
    seed=20261014;torch.manual_seed(seed);history=[]
    gradient_groups={name:0. for name in ['visual_prompts_encoder','fusion_blocks','neck','head']}
    real=torch.load(refs['val'][0]['file'],map_location='cuda',weights_only=False)
    result={'baseline':baseline,'pretrained':initial,'arms':{}}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        (args.output/'status.json').write_text(json.dumps({'status':'training','arm':'segnext','epoch':epoch,'worker_pid':os.getpid()}))
        started=time.time();losses=[];model.train()
        for step,index in enumerate(np.random.default_rng(seed+epoch).permutation(len(refs['train']))):
            sample=torch.load(refs['train'][index]['file'],map_location='cuda',weights_only=False)
            loss=dense_loss(model(sample),sample)
            if not torch.isfinite(loss):raise FloatingPointError('loss')
            optimizer.zero_grad();loss.backward()
            if epoch==0 and step<8:
                for name,param in model.model.named_parameters():
                    group=name.split('.')[0]
                    if group in gradient_groups and param.grad is not None:gradient_groups[group]+=float(param.grad.detach().abs().sum())
            norm=torch.nn.utils.clip_grad_norm_(trainable,5)
            if not torch.isfinite(norm):raise FloatingPointError('gradient')
            optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%100==0:print('TRAIN',epoch,step+1,'/',len(refs['train']),'seconds',round(time.time()-started,1),flush=True)
        training_seconds=time.time()-started;model.eval()
        summary,_=evaluate(model,refs['val'],'cuda',args.output/f'segnext_epoch_{epoch:03}_rows.json')
        with torch.no_grad():
            altered={**real,'truth':~real['truth'],'valid':~real['valid'],'target_class':torch.tensor([11],device='cuda')}
            if not torch.equal(model(real)['pixel_probability'],model(altered)['pixel_probability']):raise RuntimeError('GT-dependent inference')
        summary.update(epoch=epoch,optimizer_steps=(epoch+1)*len(refs['train']),train_loss=float(np.mean(losses)),
                       train_seconds=training_seconds,seconds=time.time()-started,trainable_parameters=sum(p.numel() for p in trainable),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated())
        summary['eligible']=(summary['bf1_2']>=baseline['bf1_2']+.01 and summary['dice']>=baseline['dice']-.002
                             and summary['micro_dice']>=baseline['micro_dice']-.002 and summary['credible_violations']<=baseline['credible_violations'])
        history.append(summary)
        result['arms']['segnext']={'history':history,'gradient_groups':gradient_groups}
        torch.save({'model':model.model.state_dict(),'config':model.config,'epoch':epoch,'summary':summary,'run':run},args.output/f'segnext_epoch_{epoch:03}.pth')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        print('EPOCH',json.dumps(summary),flush=True)
    if any(v<=0 or not np.isfinite(v) for v in gradient_groups.values()):raise RuntimeError('missing gradient group')
    frozen_after=backbone_sha(model)
    if frozen_before!=frozen_after:raise RuntimeError('frozen backbone changed')
    payload=torch.load(args.output/f'segnext_epoch_{args.epochs-1:03}.pth',map_location='cpu',weights_only=False)
    reloaded=SegNextAdapter(payload['config'],payload['model']).cuda().eval()
    with torch.no_grad():reload_error=float((model(real)['pixel_probability']-reloaded(real)['pixel_probability']).abs().max())
    if reload_error>2e-6:raise RuntimeError('checkpoint reload mismatch')
    result['arms']['segnext'].update(reload_max_error=reload_error,backbone_sha256=frozen_after,backbone_unchanged=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'status.json').write_text(json.dumps({'status':'complete','worker_pid':os.getpid()}))
    print('TRAIN_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--upstream',type=Path)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if ARTIFACT not in args.output.resolve().parents or args.epochs<1 or args.lr<=0:raise ValueError('output/budget')
    if args.smoke and args.stage=='train' and args.epochs!=1:raise ValueError('smoke epoch budget')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    (args.output/'status.json').write_text(json.dumps({'status':'initializing','stage':args.stage,'worker_pid':os.getpid()}))
    (prepare if args.stage=='prepare' else train)(args)


if __name__=='__main__':main()
