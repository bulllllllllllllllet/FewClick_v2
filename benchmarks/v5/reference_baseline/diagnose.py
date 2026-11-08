"""Readonly native-author reconstruction and fixed positive-only diagnostic."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from functools import lru_cache
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
from benchmarks.v5.support_match_study.model import DecoderData
from benchmarks.v5.paper_multiscale.data import native_context
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.wsi_quality_study.prepare import dump
from .pilot import summarize
from torchvision import transforms

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=a.output;r.mkdir(exist_ok=False);src=a.source;c=json.load(open(src/'contract.json'));weights=json.load(open(src/'weights.json'));tasks=json.load(open(src/'plans.json'));assert sha(src/'plans.json')==c['plans_sha256']
 dump(r/'stage.json',dict(status='running',worker_pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
 for v in weights['weights'].values():assert sha(v['path'])==v['sha256']
 sys.path[:0]=[str(src/'deps'),str(src/'SANSA')]
 from util.path_utils import SAM2_PATHS_CONFIG
 SAM2_PATHS_CONFIG['large']=(weights['weights']['sam2_hiera_large.pt']['path'],SAM2_PATHS_CONFIG['large'][1])
 # Import the author's actual TorchHub constructor, with local verified downloads.
 spec=importlib.util.spec_from_file_location('official_sansa_hub',src/'SANSA/hubconf.py');hub=importlib.util.module_from_spec(spec);spec.loader.exec_module(hub)
 torch.set_num_threads(2);torch.manual_seed(20261103);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=hub.sansa(pretrained=False,device='cuda').eval();state=torch.load(weights['weights']['sansa_universal.pth']['path'],map_location='cpu',weights_only=False)['model'];missing,extra=model.load_state_dict(state,strict=False);assert not extra and not any('adapter' in k for k in missing);before=tree_sha(model.state_dict());assert not model.sam.pred_obj_scores
 # Audit base initialization coverage, including ignored original occlusion heads.
 base=torch.load(weights['weights']['sam2_hiera_large.pt']['path'],map_location='cpu',weights_only=False)['model'];native=model.sam.state_dict();expected_nonadapter=[k for k in native if 'adapter' not in k];assert all(k in base and torch.equal(native[k].cpu(),base[k]) for k in expected_nonadapter)
 dump(r/'weight_acceptance.json',dict(all_expected_nonadapter_loaded=len(expected_nonadapter),base_extra_keys=sorted(set(base)-set(native)),adapter_keys=len(state),official_pred_obj_scores=model.sam.pred_obj_scores,full_state_sha256=before))
 del native,base,state
 data=DecoderData(Path(c['cache']),'match');data.rows.update(json.load(open(Path(c['source_tasks_root'])/'query_rows.json')))
 standard=transforms.Compose([transforms.Resize((640,640)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 @lru_cache(maxsize=32)
 def img(identity):return Image.fromarray(native_context(data.rows[identity],0)).resize((512,512),Image.Resampling.BILINEAR)
 old=np.load(src/'masks.npy',mmap_mode='r');results={k:[] for k in ['official_gpu3pos2neg','official_3pos2neg','official_3pos_only']};masks={k:[] for k in results};points=[];preprocess_differences=[];cpu_mask_differences=[];start=time.monotonic()
 with torch.inference_mode():
  for j,t in enumerate(tasks):
   cpu_video=torch.stack([standard(img(t['support_id'])),standard(img(t['query_id']))])[None].cuda();gpu_frames=[]
   for identity in [t['support_id'],t['query_id']]:
    raw=np.asarray(img(identity).resize((640,640),Image.Resampling.BILINEAR)).copy();x=torch.from_numpy(raw).permute(2,0,1).float().cuda()/255;gpu_frames.append((x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None])
   gpu_video=torch.stack(gpu_frames)[None];preprocess_differences.append(float((cpu_video-gpu_video).abs().max()))
   for name in results:
    video=gpu_video if name=='official_gpu3pos2neg' else cpu_video;neg=[] if name=='official_3pos_only' else t['negative_xy'];xy=torch.tensor([t['positive_xy']+neg],dtype=torch.float32,device='cuda')*640;labels=torch.tensor([[1]*len(t['positive_xy'])+[0]*len(neg)],dtype=torch.int32,device='cuda');prompt={0:{0:dict(prompt_type='point',prompt=dict(point_coords=xy,point_labels=labels))},'shots':1}
    output=model(video,prompt)['pred_masks'];assert output.shape==(2,640,640) and torch.isfinite(output).all();pred=(F.interpolate(output[1][None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0]>=0).cpu().numpy()
    if name=='official_gpu3pos2neg':assert np.array_equal(np.packbits(pred),old[j]),'official hub with original preprocessing does not match pilot'
    if name=='official_3pos2neg':cpu_mask_differences.append(int(np.count_nonzero(pred!=np.unpackbits(old[j],count=512*512).reshape(512,512).astype(bool))))
    gt,mpp=data.labels(t['query_id']);metrics=score(pred,gt==t['target'],gt!=255,mpp);assert metrics['positive']==t['query_positive_pixels'];results[name].append(dict(index=t['index'],metrics=metrics));masks[name].append(np.packbits(pred))
    support=(F.interpolate(output[0][None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0]>=0).cpu().numpy();positive=[bool(support[min(511,int(y*512)),min(511,int(x*512))]) for x,y in t['positive_xy']];negative=[not bool(support[min(511,int(y*512)),min(511,int(x*512))]) for x,y in neg];points.append(dict(index=t['index'],mode=name,positive_satisfied=positive,negative_satisfied=negative,support_foreground_fraction=float(support.mean()),query_foreground_fraction=float(pred.mean())))
   if j%10==0:dump(r/'progress.json',dict(rows=j+1,total=110,pid=os.getpid(),seconds=time.monotonic()-start));print(json.dumps(dict(rows=j+1,total=110)),flush=True)
 assert tree_sha(model.state_dict())==before
 for name in results:
  np.save(r/f'{name}_masks.npy',np.stack(masks[name]));dump(r/f'{name}_rows.json',results[name])
  for row,t,packed in zip(json.load(open(r/f'{name}_rows.json')),tasks,np.load(r/f'{name}_masks.npy',mmap_mode='r')):
   gt,mpp=data.labels(t['query_id']);pred=np.unpackbits(packed,count=512*512).reshape(512,512).astype(bool);assert score(pred,gt==t['target'],gt!=255,mpp)==row['metrics']
 dump(r/'support_clicks.json',points);summary={name:summarize(rows,tasks) for name,rows in results.items()};compliance={}
 for name in results:
  rr=[x for x in points if x['mode']==name];compliance[name]={sign:float(np.mean([v for x in rr for v in x[sign+'_satisfied']])) if any(x[sign+'_satisfied'] for x in rr) else None for sign in ['positive','negative']}
 dump(r/'report.json',dict(status='complete_accepted_readonly_diagnostic',summary=summary,support_click_compliance=compliance,official_hub_original_GPU_preprocess_masks_exact=110,official_CPU_preprocess_mask_differences=cpu_mask_differences,CPU_GPU_input_max_abs=preprocess_differences,object_presence_head_disabled_by_official=True,limitations='No target adaptation; source-only point ablation diagnostic, not checkpoint/threshold choice. Disabled presence head is a mechanistic limitation, not causally isolated explanation of all errors.'))
 dump(r/'acceptance.json',dict(status='complete_accepted',worker_pid=os.getpid(),report_sha256=sha(r/'report.json'),source_sha256=sha(__file__),plans_sha256=sha(src/'plans.json'),weights_unchanged=True,official_mask_replay_exact=110,independent_mask_rescore=330))
 dump(r/'stage.json',dict(status='complete_accepted',worker_pid=os.getpid(),rows=330))
if __name__=='__main__':main()
