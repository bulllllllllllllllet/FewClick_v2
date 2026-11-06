"""Continuous source-WSI ROI gate; indexed center GT explicitly bounded."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch,pyvips
from benchmarks.v5.support_match_study.model import DecoderData,DecoderModel
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.fair_comparison.metrics import score
from benchmarks.v5.encoder_research.continuous_data import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 plan=json.load(open(r/'wsi_grid_plan.json'));assert len(plan['plans'])==3
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=r/'continuous_wsi';out.mkdir(exist_ok=False)
 cache=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/cell_repair_rebalance_1102_210552/normalized');data=DecoderData(cache,'match');m=DecoderModel('match');ck=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/support_region_study_1106_162117/control/control/checkpoint.pt');m.load_state_dict(torch.load(ck,weights_only=True,map_location='cuda:0')['model']);m.eval();torch.cuda.reset_peak_memory_stats();rows=[];beg=time.monotonic()
 with torch.no_grad():
  for i,item in enumerate(plan['plans']):
   task=item['task'];inputs=data.model_inputs(task,'xcell','cuda:0');slide=pyvips.Image.new_from_file(item['geometry']['source_image'],access='random');acc=np.zeros((1024,1024),np.float64);weight=np.zeros_like(acc);tile_probs=[];io=0.;infer=0.;rgb_canvas=np.zeros((1024,1024,3),np.uint8)
   for tile in item['tiles']:
    start=time.monotonic();q={k:v.clone() for k,v in inputs['query'].items()};images=[]
    for s,box in enumerate(tile['boxes']):
     x,y,w,h=box;assert x+w<=slide.width and y+h<=slide.height
     crop=slide.crop(x,y,w,h).extract_band(0,n=3);arr=np.ndarray(buffer=crop.write_to_memory(),dtype=np.uint8,shape=(h,w,3)).copy();arr=np.asarray(Image.fromarray(arr).resize((512,512),Image.Resampling.BILINEAR)).copy();images.append(arr);q['image'+('' if s==0 else '_'+str(s))]=he_to_tensor(arr)[None].to('cuda:0')
    io+=time.monotonic()-start
    if tile['dx']==tile['dy']==0:
     ref=data.inputs(task['query_id'],'xcell')['image'];delta=(he_to_tensor(images[0])[None]-ref).abs().mean().item();assert delta<.06,('source crop mismatch normalized RGB',delta)
    torch.cuda.synchronize();start=time.monotonic();prob=m(**dict(inputs,query=q,local=False,shared=False))['query']['pixel_probability'][0].cpu().numpy();torch.cuda.synchronize();infer+=time.monotonic()-start
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all();tile_probs.append(prob)
    x=(tile['dx']+1)*256;y=(tile['dy']+1)*256;acc[y:y+512,x:x+512]+=prob;weight[y:y+512,x:x+512]+=1;rgb_canvas[y:y+512,x:x+512]=images[0]
    print(json.dumps(dict(stage='continuous_ROI_inference',worker_pid=os.getpid(),slide=i+1,tile=len(tile_probs),total_tiles=9)),flush=True)
   assert (weight>0).all();fused=(acc/weight).astype('float32');truth,mpp=data.labels(task['query_id']);center=fused[256:768,256:768]>=.5
   metrics=score(center,truth==task['target'],truth!=255,mpp)
   np.savez_compressed(out/f'roi_{i}.npz',probability=fused,tile_probabilities=np.stack(tile_probs),weight=weight.astype('uint8'))
   overlay=rgb_canvas.copy();mask=fused>=.5;overlay[mask]=(.6*overlay[mask]+.4*np.array([255,0,0])).astype('uint8');Image.fromarray(np.concatenate([rgb_canvas,overlay],axis=1)).save(out/f'roi_{i}.png')
   # Disk replay: recompute fusion and center scores from tile probabilities.
   with np.load(out/f'roi_{i}.npz') as z:
    a2=np.zeros_like(acc);w2=np.zeros_like(weight)
    for t,prob in zip(item['tiles'],z['tile_probabilities']):
     x=(t['dx']+1)*256;y=(t['dy']+1)*256;a2[y:y+512,x:x+512]+=prob;w2[y:y+512,x:x+512]+=1
    assert np.array_equal((a2/w2).astype('float32'),z['probability']);assert score(z['probability'][256:768,256:768]>=.5,truth==task['target'],truth!=255,mpp)==metrics
   rows.append(dict(slide=item['slide'],target=task['target'],tiles=9,center_only_metrics=metrics,source_center_rgb_mean_abs_normalized=delta,source_IO_preprocessing_seconds=io,inference_seconds=infer,output_sha256=sha(out/f'roi_{i}.npz')))
 (out/'report.json').write_text(json.dumps(dict(status='accepted_continuous_ROI_engineering_gate',rows=rows,total_seconds=time.monotonic()-beg,peak_GPU_MiB=torch.cuda.max_memory_allocated()/1024**2,checkpoint_sha256=sha(ck),query_human_clicks=0,limitation='3 continuous ROIs, not full WSIs; GT available only for indexed centers; IO measured after model load and may use OS cache; full ROI accuracy and complete cold/cached interaction benchmarks remain pending'),indent=2))
 html='<meta charset="utf-8"><h1>连续WSI区域拼接门控</h1><p>每片9窗口，50%重叠；先融合概率再阈值。只在中心已有标注评分，周边红色预测尚无完整GT，不是全片质量结论。</p>'
 for i,row in enumerate(rows):html+=f'<p>{row["slide"]}，目标{row["target"]}</p><img width="1200" src="roi_{i}.png">'
 (out/'index.html').write_text(html)
if __name__=='__main__':main()
