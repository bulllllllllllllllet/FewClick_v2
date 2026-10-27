"""Actual same-WSI image-only/prototype/closed-reference disk-cache gates."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import pyvips
import numpy as np
from PIL import Image
import torch

from .cache_gate import raw_images,load_cache
from benchmarks.v5.external_evaluation.evaluate_adapted import load
from benchmarks.v5.heldout_transfer.features import sam_features,wsi_features
from benchmarks.v5.transfer_comparison.prototypes import dense_prototype,region_prototype
from benchmarks.v5.fair_comparison.protocol import nested


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def check(test,message):
    if not test:raise ValueError(message)


def raw(images):
    values=raw_images(images)
    for scale,image in enumerate(images):values['rgb'+('' if scale==0 else f'_{scale}')]=torch.from_numpy(image.copy()).permute(2,0,1)[None]
    return values


def encode(model,method,values):
    if method=='fewclick_m3d_image':return model.core.encode_image(values)
    if method=='sam_ft':return sam_features(model,values)
    if method=='wsi_sam_ft':return wsi_features(model,values)
    if method=='closed_reference':return model.network(values['image'])['out']
    raise ValueError(method)


def prompts(plan,budget):
    plan=nested(plan,budget);p={}
    for sign in ('positive','negative'):
        p[sign+'_xy']=torch.tensor([plan[sign+'_xy']],dtype=torch.float32);p[sign+'_mask']=torch.ones(1,len(plan[sign+'_xy']),dtype=torch.bool)
    p['prompt_size_id']=torch.ones(1,dtype=torch.long)
    return p


def decode(model,method,source,query,p):
    if method=='fewclick_m3d_image':
        task=model.core.support_task(source,p);out,maps=model.core.decode(query,task,p,False);check(not maps.count_nonzero(),'remote maps must be zero')
        return {'graph_dense':out['pixel_probability'][0], 'region_prototype':region_prototype(source,query,p)['pixel_probability'][0]}
    return {'feature_prototype':dense_prototype(source,query,p)['pixel_probability'][0]}


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);pid=os.getpid();start=time.perf_counter();torch.set_num_threads(4)
    def dump(n,d):(a.output/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        check(not torch.cuda.is_available(),'explicit CPU gate')
        accepted=read(a.acceptance/'summary.json');check(accepted['status']=='accepted_source_band_query_protocol' and accepted['protocol_summary_sha256']==sha(a.protocol/'summary.json'),'accepted actual source/query protocol')
        development=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208');training=development/'adaptation_1025_131451'
        inputs=read(a.protocol/'query_inputs.json');maximum=read(a.protocol/'support_tasks_maximum.json');sources={};queries={};cases=[];identities={}
        for index,row in enumerate(inputs):
            sid=row['wsi_id'];plans=[next(p for p in maximum if p['wsi_id']==sid and p['target']==0 and p['prompt_group']==0)]
            if index==0:plans+=[next(p for p in maximum if p['wsi_id']==sid and p['target']==10 and p['prompt_group']==0)]
            for plan in plans:
                record=next(p for p in row['support_inputs'] if p['support_id']==plan['support_id'])
                images=[]
                for v in record['images']:
                    check(sha(v['path'])==v['sha256'],'native source RGB hash');images.append(np.asarray(Image.open(v['path']).convert('RGB')).copy())
                sources[plan['support_id']]=raw(images);identities[plan['support_id']]=record
            grid=[json.loads(line) for line in Path(row['grid']).open()];native=pyvips.Image.new_from_file(row['source_wsi'],access='random')
            for number in (len(grid)//2,len(grid)-1):
                tile=grid[number];identity=sid+':query:'+str(tile['number']);images=[];paths=[]
                for scale,b in enumerate(tile['native_boxes']):
                    check(tile['native_boxes'][2][0]>=row['source_cut_native']+row['gap_native'],'strict remote context')
                    crop=native.crop(*b).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(b[3],b[2],3)
                    im=Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR);p=a.output/(identity.replace(':','_')+f'_scale{scale}.png');im.save(p);images.append(np.asarray(im).copy());paths.append(dict(path=str(p),sha256=sha(p),native_box=b))
                queries[identity]=raw(images);identities[identity]=dict(source_wsi=row['source_wsi'],images=paths)
                for plan in plans:cases.append(dict(source=plan['support_id'],query=identity,plan=plan))
        paths=[Path(__file__),Path('benchmarks/v5/wsi_global/cache_gate.py'),Path('benchmarks/v5/paper_multiscale/model.py'),Path('benchmarks/v5/fair_comparison/adapted_models.py'),Path('benchmarks/v5/external_evaluation/evaluate_adapted.py'),Path('benchmarks/v5/heldout_transfer/features.py'),Path('benchmarks/v5/transfer_comparison/prototypes.py')]
        source_sha={str(p):sha(p) for p in paths};dump('contract.json',dict(worker_pid=pid,source_sha256=source_sha,protocol_sha256=sha(a.protocol/'summary.json'),acceptance_sha256=sha(a.acceptance/'summary.json'),query_GT_access=False,device='CPU4 FP32',scope='NoCell/prototype/closed12 gate, not full-cell or quality scores'))
        results=[];outputs=[];cache_artifacts=[]
        for method in ('fewclick_m3d_image','sam_ft','wsi_sam_ft','closed_reference'):
            folder=a.output/method;folder.mkdir();best=read(training/f'train_{method}_1025_131451/best_checkpoint.json');done=read(training/f'train_{method}_1025_131451/summary.json')
            check(done['status']=='complete' and done['step']==10000 and done['training_consumed_sha256']=='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe','accepted full-label40k training')
            check(sha(best['path'])==best['checkpoint_sha256'],'selected development checkpoint')
            model=load(method,Path(best['path']),torch.device('cpu'),sha(development/'tasks_1025_125400/contract.json'));model.eval();cached={};count=0
            encoder_calls=[0]
            def hook(module,args):encoder_calls[0]+=1
            modules=[e.backbone for e in model.core.encoders] if method=='fewclick_m3d_image' else [model.network.backbone] if method=='closed_reference' else [model.sam.image_encoder]
            hooks=[module.register_forward_pre_hook(hook) for module in modules]
            selected_raw=queries if method=='closed_reference' else sources|queries
            with torch.inference_mode():
                for identity,values in selected_raw.items():
                    native=encode(model,method,values);count+=1;p=folder/(identity.replace(':','_')+'.pth')
                    torch.save(dict(encoded=native,identity=identities[identity],checkpoint_sha256=best['checkpoint_sha256']),p)
                    disk=torch.load(p,map_location='cpu',weights_only=False);check(disk['identity']==identities[identity] and disk['checkpoint_sha256']==best['checkpoint_sha256'],'disk-cache identity')
                    cached[identity]=load_cache(p,best['checkpoint_sha256'],identities[identity])
                    cache_artifacts.append(dict(method=method,identity=identity,path=str(p),sha256=sha(p)))
                offline_encoder_calls=encoder_calls[0];errors=[];formula_differences=[]
                identity=next(iter(cached));path=folder/(identity.replace(':','_')+'.pth');rejected=0
                for weight,metadata in [('0'*64,identities[identity]),(best['checkpoint_sha256'],identities[identity]|{'wrong_input_probe':True})]:
                    try:load_cache(path,weight,metadata)
                    except ValueError:rejected+=1
                check(rejected==2,'actual cache consumer wrong weight/input rejection')
                if method=='closed_reference':
                    for i,(identity,values) in enumerate(queries.items()):
                        before=encoder_calls[0];probability=cached[identity].softmax(1)
                        check(encoder_calls[0]==before,'closed cached probability has no encoder call')
                        direct=model(values,{},dataset='crc',target=0)
                        error=float((cached[identity]-direct['class_logits']).abs().max())
                        reconstructed=(cached[identity][:,0]-torch.logsumexp(cached[identity][:,1:],1)).sigmoid()
                        check(error==0 and torch.equal(reconstructed,direct['pixel_probability']) and torch.equal(cached[identity].argmax(1)==0,direct['categorical_prediction']),'closed12 identical logits/formula/categorical identity')
                        formula_differences.append(float((probability[:,0]-direct['pixel_probability']).abs().max()))
                        p=folder/f'probability_{i:03d}.npy';np.save(p,probability[0].numpy(),allow_pickle=False);outputs.append(dict(method=method,query=identity,path=str(p),sha256=sha(p),kind='12class_softmax'));errors.append(error)
                else:
                    for i,case in enumerate(cases):
                        for budget in ('3+2','1+1','3+1','5+1'):
                            p=prompts(case['plan'],budget);before=encoder_calls[0]
                            values=decode(model,method,cached[case['source']],cached[case['query']],p)
                            check(encoder_calls[0]==before,'actual encoder hooks zero during online decode')
                            for name,value in values.items():
                                check(value.shape==(512,512) and torch.isfinite(value).all() and value.min()>=0 and value.max()<=1,'finite native query probability')
                                path=folder/f'probability_{i:03d}_{budget}_{name}.npy';np.save(path,value.numpy(),allow_pickle=False);check(np.array_equal(np.load(path),value.numpy()),'saved probability identity')
                                outputs.append(dict(method=method,variant=name,case=i,budget=budget,path=str(path),sha256=sha(path)))
                            if i in (0,len(cases)-2) and budget=='3+2':
                                if method=='fewclick_m3d_image':
                                    direct=model.core(sources[case['source']],queries[case['query']],p,False,False,with_support=False);check(not direct['query_prompt_maps'].count_nonzero(),'native forward remote maps zero');expected=direct['query']['pixel_probability'][0]
                                    actual=values['graph_dense']
                                else:
                                    direct_source=encode(model,method,sources[case['source']]);direct_query=encode(model,method,queries[case['query']]);expected=decode(model,method,direct_source,direct_query,p)['feature_prototype'];actual=values['feature_prototype']
                                error=float((actual-expected).abs().max());check(error==0 and torch.equal(actual>=.5,expected>=.5),'fresh native RGB path versus disk encoded cache');errors.append(error)
            for handle in hooks:handle.remove()
            results.append(dict(method=method,checkpoint_sha256=best['checkpoint_sha256'],selected_step=best['step'],cache_files=len(cached),native_errors=errors,online_query_encode_calls=0,offline_encoder_forward_calls=offline_encoder_calls,verification_encoder_forward_calls=encoder_calls[0]-offline_encoder_calls,negative_cache_cases=rejected,closed_softmax_vs_binary_formula_differences=formula_differences))
            dump('status.json',dict(status='running',worker_pid=pid,completed_methods=len(results),total_methods=4));print(json.dumps(results[-1]),flush=True)
            dump('outputs.json',outputs);dump('cache_manifest.json',cache_artifacts)
            del model,cached;gc.collect()
        dump('outputs.json',outputs);dump('input_identities.json',identities);dump('cases.json',cases)
        result=dict(status='passed_WSI_native_remote_cache_gate',worker_pid=pid,methods=results,outputs=len(outputs),cases=len(cases),source_sha256=source_sha,output_manifest_sha256=sha(a.output/'outputs.json'),seconds=time.perf_counter()-start,
            query_GT_access=False,scope='actual same-WSI source/remote image-only native, custom encoder prototype and privileged closed12 disk-cache integration; no accuracy or full-cell acceptance')
        dump('summary.json',result);dump('status.json',dict(status=result['status'],worker_pid=pid));print(json.dumps(dict(status=result['status'],seconds=result['seconds'])),flush=True)
    except Exception:
        dump('status.json',dict(status='failed',worker_pid=pid,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('protocol','acceptance','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
