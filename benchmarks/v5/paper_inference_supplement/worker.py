import argparse, json, os, time, gc, subprocess
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from .common import Inputs,load,encode,decode,clear,tensor_bytes,verify,prompts
from benchmarks.v5.whole_slide_once.worker import configure
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.sam_current_fair.resources import acquire


def full(original,x,q,p,local):
    return original(support=x,query=q,prompts=p,local=local,shared=local,with_support=False)['query']['pixel_probability']


@torch.no_grad()
def gate(model,original,method,data,loader,local_tasks,remote_tasks,out):
    initial=tree_sha(original.state_dict()); checks=[]
    selected=[next(t for t in local_tasks if t['dataset']==d) for d in ['crc','tiger']]
    selected += [next(t for t in remote_tasks if t['config']==config and t['size_band']==band) for config,band in [('3+2','small'),('1+1','large'),('3+1','absent'),('jitter2px','small'),('jitter5px','large')]]
    for t in selected:
        local=t.get('local',False);x=loader.get(t['support_id']);q=x if local else loader.get(t['query_id']);p=prompts(t)
        # Match the previous full input path, including legacy cell tensors, exactly.
        def legacy(identity,images):
            v={k:a.cuda() for k,a in data.inputs(data.rows[identity].get('template_id',identity),'xcell').items()}
            for i in range(3):
                k='image'+('' if i==0 else '_'+str(i));v[k]=images[k]
            return v
        sx=legacy(t['support_id'],x);qx=sx if local else legacy(t['query_id'],q)
        ref=full(original,sx,qx,p,local);se=encode(model,x,method);qe=se if local else encode(model,q,method)
        actual=decode(model,se,qe,p,method,local)
        assert torch.isfinite(actual).all() and actual.shape==(1,512,512)
        assert torch.equal(ref,actual),(method,t.get('config','local'),float((ref-actual).abs().max()))
        repeat=decode(model,se,qe,p,method,local);assert torch.equal(actual,repeat)
        checks.append(dict(dataset=t['dataset'],local=local,config=t.get('config'),cached_full_exact=True,empty_cell_input_exact=True,repeat_exact=True))
        clear(model)
    assert tree_sha(original.state_dict())==initial
    dump(out/'gate.json',dict(status='accepted_exact_inference',checks=checks,weights_unchanged=True,state_sha256=initial))
    return initial


@torch.no_grad()
def quality(model,method,data,loader,tasks,out):
    @lru_cache(maxsize=8)
    def encoded(identity):return encode(model,loader.get(identity),method)
    result=[];masks=[];start=time.monotonic();last=None
    for t in tasks:
        pair=(t['support_id'],t['query_id'])
        if pair!=last:clear(model);last=pair
        local=t.get('local',False);p=prompts(t);se=encoded(t['support_id']);qe=se if local else encoded(t['query_id'])
        prob=decode(model,se,qe,p,method,local);assert torch.isfinite(prob).all()
        mask=prob[0].cpu().numpy()>=.5
        gt,mpp=data.labels(t['query_id']);m=score(mask,gt==t['target'],gt!=255,mpp);assert m['positive']==t['query_positive_pixels']
        result.append(dict(index=t['index'],metrics=m));masks.append(np.packbits(mask))
        if len(result)%25==0 or len(result)==1:
            dump(out/'progress.json',dict(pid=os.getpid(),status='inference_running',done=len(result),total=len(tasks),seconds=time.monotonic()-start));print(json.dumps({'done':len(result),'total':len(tasks)}),flush=True)
    dump(out/'rows.json',result);np.save(out/'masks.npy',np.stack(masks));encoded.cache_clear();clear(model)
    stored=np.load(out/'masks.npy',mmap_mode='r')
    dump(out/'progress.json',dict(pid=os.getpid(),status='disk_scoring',done=len(tasks),total=len(tasks)))
    for row,t,packed in zip(json.loads((out/'rows.json').read_text()),tasks,stored):
        assert row['index']==t['index'];gt,mpp=data.labels(t['query_id']);mask=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool)
        assert score(mask,gt==t['target'],gt!=255,mpp)==row['metrics']
    return dict(rows=len(result),rows_sha256=sha(out/'rows.json'),masks_sha256=sha(out/'masks.npy'),disk_rescored=len(result),seconds=time.monotonic()-start)


def sync():torch.cuda.synchronize()


@torch.no_grad()
def timing(model,original,method,loader,tasks,out):
    # Model loading, gates and warmup excluded; no application-level input/feature reuse between cases.
    for t in tasks[:3]:
        x=loader.get(t['support_id']);q=loader.get(t['query_id']);se=encode(model,x,method);qe=encode(model,q,method)
        decode(model,se,qe,prompts(t),method);clear(model)
    del x,q,se,qe;gc.collect();torch.cuda.empty_cache();sync()
    rows=[]
    for i,t in enumerate(tasks):
        clear(model);gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        x=loader.get(t['support_id']);q=loader.get(t['query_id']);p=prompts(t)
        sync();input_end=time.perf_counter()
        se=encode(model,x,method);qe=encode(model,q,method);sync();encode_end=time.perf_counter()
        probability=decode(model,se,qe,p,method);sync();end=time.perf_counter()
        cold_peak=torch.cuda.max_memory_allocated();cold_reserved=torch.cuda.max_memory_reserved()
        assert torch.isfinite(probability).all()
        cache_bytes=tensor_bytes([se,qe,getattr(model,'support_memory',{})])
        changed=dict(t,positive_xy=t['positive_xy'][:1],negative_xy=t['negative_xy'][:1]);p2=prompts(changed)
        torch.cuda.reset_peak_memory_stats();sync();warm_start=time.perf_counter()
        clear(model) # Rebuild target-dependent memory for the changed click set.
        updated=decode(model,se,qe,p2,method);sync();warm_end=time.perf_counter()
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        assert torch.isfinite(updated).all()
        if i<2:assert torch.equal(updated,full(original,x,q,p2,False))
        rows.append(dict(index=t['index'],input_seconds=input_end-start,encode_seconds=encode_end-input_end,first_decode_seconds=end-encode_end,first_total_seconds=end-start,changed_prompt_seconds=warm_end-warm_start,cache_tensor_bytes=cache_bytes,first_peak_allocated_bytes=cold_peak,first_peak_reserved_bytes=cold_reserved,update_peak_allocated_bytes=peak,update_peak_reserved_bytes=reserved))
        clear(model);del x,q,p,p2,se,qe,probability,updated
        if (i+1)%10==0:
            dump(out/'progress.json',dict(pid=os.getpid(),status='timing_running',done=i+1,total=100));print(json.dumps({'timing':i+1,'total':100}),flush=True)
    assert len(rows)==100
    dump(out/'timing_rows.json',rows)
    columns=[k for k in rows[0] if k!='index'];summary={k:float(np.mean([r[k] for r in rows])) for k in columns}
    summary.update(n=100,statistic='arithmetic_mean',device=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],scope='remote one support/one query task; initial inputs+encoding+first prediction and changed1+1 prompts using same image features; tensor storage cache excludes model weights; peak allocated includes model weights')
    dump(out/'timing_summary.json',summary)
    return dict(rows=100,timing_rows_sha256=sha(out/'timing_rows.json'),timing_summary_sha256=sha(out/'timing_summary.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--method',required=True);parser.add_argument('--stage',choices=['local','robust','timing'],required=True);a=parser.parse_args()
    c=verify(a.root);out=a.root/f'{a.stage}_{a.method}';out.mkdir()
    dump(out/'worker.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],status='waiting_resource',stage=a.stage,method=a.method))
    lease=acquire(c,int(os.environ['CUDA_VISIBLE_DEVICES']),a.stage+'_'+a.method)
    try:
        configure();ck=c['checkpoints'][a.method];assert sha(ck['path'])==ck['sha256']
        local=json.loads((a.root/'local_tasks.json').read_text());remote=json.loads((a.root/'robust_tasks.json').read_text());tasks=json.loads((a.root/(a.stage+'_tasks.json')).read_text())
        data=DecoderData(c['cache'],'match');data.rows.update(json.loads((a.root/'rows.json').read_text()));loader=Inputs(data.rows)
        model,original=load(a.method,ck['path']);dump(out/'worker.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],status='gate_running',stage=a.stage,method=a.method))
        state=gate(model,original,a.method,data,loader,local,remote,out)
        dump(out/'gpu_before.json',dict(snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)))
        result=timing(model,original,a.method,loader,tasks,out) if a.stage=='timing' else quality(model,a.method,data,loader,tasks,out)
        assert tree_sha(original.state_dict())==state
        dump(out/'input_hashes.json',loader.hashes)
        dump(out/'acceptance.json',dict(status='complete_accepted',stage=a.stage,method=a.method,pid=os.getpid(),checkpoint_sha256=ck['sha256'],contract_sha256=sha(a.root/'contract.json'),tasks_sha256=sha(a.root/(a.stage+'_tasks.json')),input_hashes_sha256=sha(out/'input_hashes.json'),weights_unchanged=True,**result))
        print(json.dumps({'status':'complete_accepted','stage':a.stage,'method':a.method}),flush=True)
    except Exception as error:
        dump(out/'failure.json',dict(status='failed',error=repr(error)));raise
    finally:lease.close()
if __name__=='__main__':main()
