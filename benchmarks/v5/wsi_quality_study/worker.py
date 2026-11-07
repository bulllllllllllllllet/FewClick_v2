"""One frozen model / shard; cache equivalence gate before continuous inference."""
import argparse,json,time,os
from pathlib import Path
import numpy as np,torch,pyvips
from PIL import Image,ImageDraw
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.paper_multiscale.data import native_context
from .prepare import dump

def encode(m,x,method):return m.encode(x if method=='sam' else m.base.transform(x))
def decode(m,se,qe,prompts,method):
    if method=='sam':return m.decode(se,qe,prompts,False)['pixel_probability']
    task=m.base.core.support_task(se,prompts);out=m.base.core.decode(qe,task,prompts,False)[0]
    if m.pixel is not None and m.pixel_enabled:out=m.pixel(se['pixel_embedding'],qe['pixel_embedding'],prompts,out)
    return out['pixel_probability']
def timing(fn):
    torch.cuda.synchronize();t=time.monotonic();x=fn();torch.cuda.synchronize();return x,time.monotonic()-t

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--method',choices=['full','no_match','sam'],required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--gate',action='store_true');a=p.parse_args();r=a.root
    c=json.loads((r/'contract.json').read_text());assert sha(r/'plan.json')==c['plan_sha256'];assert sha(r/'coordinate_acceptance.json')==c['coordinate_sha256']
    for path,h in c['sources'].items():assert sha(path)==h,path
    ck=c['checkpoints'][a.method];assert sha(ck['path'])==ck['sha256']
    plans=json.loads((r/'plan.json').read_text())['plans'];plans=plans[:1] if a.gate else [p for p in plans if p['index']%2==a.shard]
    out=r/(f'gate_{a.method}' if a.gate else f'{a.method}_{a.shard}');out.mkdir(exist_ok=False)
    dump(out/'worker.json',dict(pid=os.getpid(),physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],method=a.method,shard=a.shard,status='loading',plan_sha256=c['plan_sha256']))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=DecoderData(Path(c['cache']),'match')
    if a.method=='sam':
        from benchmarks.v5.sam_prior_repair.model import DecoderModel
        m=DecoderModel('sam_native')
    elif a.method=='no_match':
        from benchmarks.v5.paper_completion.model import DecoderModel
        m=DecoderModel('no_match')
    else:
        from benchmarks.v5.support_match_study.model import DecoderModel
        m=DecoderModel('match')
    m.load_state_dict(torch.load(ck['path'],map_location='cuda:0',weights_only=True)['model']);m.eval();torch.cuda.reset_peak_memory_stats();rows=[]
    with torch.no_grad():
        for item in plans:
            i=item['index'];task=item['task'];inputs=data.model_inputs(task,'xcell','cuda:0');assert not inputs['local'] and not inputs['shared']
            se,support_s=timing(lambda:encode(m,inputs['support'],a.method));slide=pyvips.Image.new_from_file(item['geometry']['source_image'],access='random');n=item['size'];acc=np.zeros((n,n),np.float64);weight=np.zeros((n,n),np.uint8);canvas=np.zeros((n,n,3),np.uint8);probs=[];times=[];gate_delta=None
            for j,tile in enumerate(item['tiles']):
                t=time.monotonic();q=dict(inputs['query']);images=[]
                for scale,box in enumerate(tile['boxes']):
                    crop=slide.crop(*box).extract_band(0,n=3);rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(crop.height,crop.width,3)
                    rgb=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR)).copy();images.append(rgb);q['image'+('' if scale==0 else '_'+str(scale))]=he_to_tensor(rgb)[None].to('cuda:0')
                io=time.monotonic()-t
                if tile['dx']==tile['dy']==0:
                    for scale in range(3):assert np.array_equal(images[scale],np.asarray(Image.fromarray(native_context(data.rows[task['query_id']],scale)).resize((512,512),Image.Resampling.BILINEAR)))
                qe,enc_s=timing(lambda:encode(m,q,a.method));prob,dec_s=timing(lambda:decode(m,se,qe,inputs['prompts'],a.method))
                if j==0:
                    reference=m(**dict(inputs,query=q,local=False,shared=False,with_support=False))['query']['pixel_probability'];gate_delta=float((reference-prob).abs().max());assert torch.equal(reference,prob),('cached path differs',gate_delta)
                    repeats=[]
                    for _ in range(3):
                        replay,dt=timing(lambda:decode(m,se,qe,inputs['prompts'],a.method));assert torch.equal(replay,prob);repeats.append(dt)
                    # True point update using permitted support prefix only, not query annotations.
                    prompts={k:v.clone() for k,v in inputs['prompts'].items()}
                    prompts['positive_mask'][:,1:]=False;prompts['negative_mask'][:,1:]=False
                    updated,update_s=timing(lambda:decode(m,se,qe,prompts,a.method))
                    original_update=m(**dict(inputs,query=q,prompts=prompts,local=False,shared=False,with_support=False))['query']['pixel_probability'];assert torch.equal(updated,original_update)
                prob=prob[0].cpu().numpy();assert prob.shape==(512,512) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all();probs.append(prob)
                x=(tile['dx']+item['radius'])*256;y=(tile['dy']+item['radius'])*256;acc[y:y+512,x:x+512]+=prob;weight[y:y+512,x:x+512]+=1;canvas[y:y+512,x:x+512]=images[0]
                times.append(dict(io_seconds=io,encode_seconds=enc_s,cached_decode_seconds=dec_s))
                print(json.dumps(dict(stage='inference',pid=os.getpid(),method=a.method,roi=i,tile=j+1,total_tiles=len(item['tiles']))),flush=True)
                if a.gate:break
            if a.gate:
                dump(out/'acceptance.json',dict(status='accepted_real_cached_forward_and_prompt_update',max_delta=gate_delta,query_human_clicks=0,checkpoint_sha256=ck['sha256'],plan_sha256=c['plan_sha256']));return
            assert (weight>0).all();fused=(acc/weight).astype(np.float32)
            np.savez_compressed(out/f'roi_{i:03}.npz',probability=fused,tile_probabilities=np.stack(probs),weight=weight)
            Image.fromarray(canvas).save(out/f'roi_{i:03}_rgb.png')
            rgb=inputs['support']['image'][0].permute(1,2,0).cpu().numpy();rgb=((rgb*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255).round().clip(0,255).astype(np.uint8)
            support=Image.fromarray(rgb);draw=ImageDraw.Draw(support)
            for sign,color in [('positive','lime'),('negative','magenta')]:
                for px,py in task[sign+'_xy']:draw.ellipse((px*512-4,py*512-4,px*512+4,py*512+4),outline=color,width=2)
            support.save(out/f'roi_{i:03}_support.png')
            row=dict(index=i,slide=item['slide'],target=task['target'],checkpoint_sha256=ck['sha256'],output_sha256=sha(out/f'roi_{i:03}.npz'),cached_forward_exact=True,support_encode_seconds=support_s,tile_times=times,cached_repeat_seconds=repeats,cached_support_prefix_update_seconds=update_s,peak_GPU_MiB=torch.cuda.max_memory_allocated()/1024**2)
            rows.append(row);dump(out/'rows.json',rows)
    dump(out/'completion.json',dict(status='inference_complete_pending_independent_acceptance',rows=len(rows),indices=[v['index'] for v in rows],rows_sha256=sha(out/'rows.json'),plan_sha256=c['plan_sha256'],pid=os.getpid()))
if __name__=='__main__':main()
