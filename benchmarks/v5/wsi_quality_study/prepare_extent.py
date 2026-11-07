"""Register geometry before label access; replay documented source-window mapping."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,pyvips,yaml
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.paper_multiscale.data import SOURCES,intersects
BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
CHECKPOINTS={'full':BASE/'support_region_study_1106_162117/control/control/checkpoint.pt',
 'no_match':BASE/'paper_completion_dirfix_1106_193345/no_match/control/checkpoint.pt',
 'sam':BASE/'sam_budget2000_1105_120404/checkpoint.pt'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def labels_for(item,box,size):
    """Coordinates derived from declared image/GT windows, never fitted to GT."""
    gt=pyvips.Image.new_from_file(item['annotation'],access='random');sx,sy,tx,ty=item['gt_affine']
    x,y,w,h=box;xs=np.floor((x+(np.arange(size)+.5)*w/size)*sx+tx).astype(int);ys=np.floor((y+(np.arange(size)+.5)*h/size)*sy+ty).astype(int)
    assert 0<=xs.min()<=xs.max()<gt.width and 0<=ys.min()<=ys.max()<gt.height
    gx,gy=int(xs.min()),int(ys.min());crop=gt.crop(gx,gy,int(xs.max()-gx+1),int(ys.max()-gy+1)).extract_band(0,n=3)
    rgb=np.frombuffer(crop.write_to_memory(),np.uint8).reshape(crop.height,crop.width,3)
    rgb=rgb[(ys-gy)[:,None],(xs-gx)[None,:]];lab=np.full((size,size),255,np.uint8)
    palette=yaml.safe_load(Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml').read_text())['data']['class_map']
    for c in palette:lab[(rgb==c['rgb']).all(-1)]=c['id']
    return lab

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
    catalog,rows=load_cell_catalog(TASKS);sources=pd.read_parquet(SOURCES).set_index('source_id').to_dict('index');train=defaultdict(list)
    for row in rows.values():
        if row['split']=='train':train[row['space_key']].append(json.loads(row['multiscale_json'])['context_um'])
    protocol=BASE/'p3_task_protocol_1103_003729/evaluation.jsonl'
    prior=BASE/'wsi_quality_1107_190254_coordfix'
    previous=json.loads((prior/'plan.json').read_text())['plans']
    tasks=[v['task'] for v in previous]
    plans=[];seen=set();rejected=[]
    for task in tasks:
        row=rows[task['query_id']];slide=row['space_key']
        if slide in seen:continue
        g=json.loads(row['multiscale_json']);s=rows[task['support_id']];sg=json.loads(s['multiscale_json']);im=pyvips.Image.new_from_file(g['source_image'],access='random')
        selected=None
        for radius in [8,6,4,2,1]:
            tiles=[];reason=None
            for dy in range(-radius,radius+1):
                for dx in range(-radius,radius+1):
                    ox=dx*g['native_boxes'][0][2]//2;oy=dy*g['native_boxes'][0][3]//2
                    boxes=[[x+ox,y+oy,w,h] for x,y,w,h in g['native_boxes']]
                    x,y,w,h=boxes[-1];mx,my=g['native_mpp'];foot=[x*mx,y*my,(x+w)*mx,(y+h)*my]
                    if x<0 or y<0 or x+w>im.width or y+h>im.height:reason='source_boundary'
                    elif any(intersects(foot,b) for b in train[slide]):reason='train_4x_overlap'
                    elif s['space_key']==slide and intersects(foot,sg['context_um']):reason='support_4x_overlap'
                    if reason:break
                    tiles.append(dict(dx=dx,dy=dy,boxes=boxes,footprint_um=foot))
                if reason:break
            if reason:rejected.append(dict(query_id=task['query_id'],support_id=task['support_id'],radius=radius,reason=reason));continue
            selected=(radius,tiles);break
        if selected is None:continue
        radius,tiles=selected;source=sources[row['source_id']];e=json.loads(source['extra_json']);ix,iy,iw,ih=e['image_window'];gx,gy,gw,gh=e['gt_window'];sx,sy=gw/iw,gh/ih;affine=[sx,sy,gx-ix*sx,gy-iy*sy]
        gt_header=pyvips.Image.new_from_file(source['annotation'],access='random');global_affine=[gt_header.width/im.width,gt_header.height/im.height,0,0]
        # Replay original build_10x_patch_index rounding, not a fitted affine.
        for ss in sources.values():
            if ss['dataset']=='crc' and ss['image']==source['image']:
                ee=json.loads(ss['extra_json']);a,b,c,d=ee['image_window'];u,v,j,k=ee['gt_window'];assert [round(u/global_affine[0]),round(v/global_affine[1]),round(j/global_affine[0]),round(k/global_affine[1])]==[a,b,c,d], ('index provenance mismatch',ss['source_id'])
        fine=tiles[0]['boxes'][0];size=512+radius*512;extent=[fine[0],fine[1],fine[2]*(radius+1),fine[3]*(radius+1)]
        plans.append(dict(index=len(plans),slide=slide,patient_id=row['patient_id'],task=task,geometry=g,tiles=tiles,radius=radius,size=size,extent=extent,annotation=source['annotation'],gt_affine=global_affine,historical_local_affine=affine,source_windows=e,support_slide=s['space_key'],support_context=sg['context_um'],mpp=[fine[2]*g['native_mpp'][0]/512,fine[3]*g['native_mpp'][1]/512]))
        seen.add(slide)
        if len(plans)==30:break
    assert len(plans)>=3
    dump(r/'plan.json',dict(status='frozen_before_labels_and_predictions',plans=plans,rejections=rejected,selection='SHA order; first safe task per slide, up to30; same30tasks; radius8/6/4/2/1 by geometry only',protocol_sha256=sha(protocol),catalog_sha256=sha(catalog/'tiles.parquet'),sources_sha256=sha(SOURCES)))
    audits=[]
    for item in plans:
        i=item['index'];old=np.asarray(Image.open(rows[item['task']['query_id']]['labels']));center=labels_for(dict(item,gt_affine=item['historical_local_affine']),item['geometry']['native_boxes'][0],512)
        assert np.array_equal(old,center),('source-window center replay failed',i,float((old==center).mean()))
        lab=labels_for(item,item['extent'],item['size']);offset=item['radius']*256;global_center=labels_for(item,item['geometry']['native_boxes'][0],512);assert np.array_equal(global_center,lab[offset:offset+512,offset:offset+512])
        path=r/f'labels_{i:03}.png';Image.fromarray(lab).save(path)
        audits.append(dict(index=i,historical_center_exact=True,global_vs_historical_center_agreement=float((global_center==old).mean()),label_sha256=sha(path),annotation_sha256=sha(item['annotation']),source_image_sha256=sha(item['geometry']['source_image']),valid_fraction=float((lab!=255).mean())))
        print(json.dumps(dict(stage='coordinate_replay',index=i,slides=len(plans),center_exact=True)),flush=True)
    dump(r/'coordinate_acceptance.json',dict(status='accepted_documented_source_window_mapping',rows=audits,limitation='historical annotation provenance retained; not independent expert registration validation; global dimension ratio now traced to original index-generation code; per-patch rounding causes recorded historical center differences; no fitting to GT'))
    source_paths=list(Path('benchmarks/v5/wsi_quality_study').glob('*.py'))+[Path('benchmarks/v5/support_match_study/model.py'),Path('benchmarks/v5/paper_completion/model.py'),Path('benchmarks/v5/sam_prior_repair/model.py')]
    dump(r/'contract.json',dict(plan_sha256=sha(r/'plan.json'),coordinate_sha256=sha(r/'coordinate_acceptance.json'),checkpoints={k:dict(path=str(v),sha256=sha(v)) for k,v in CHECKPOINTS.items()},sources={str(p):sha(p) for p in source_paths},cache=str(BASE/'cell_repair_rebalance_1102_210552/normalized'),threshold=.5,query_human_clicks=0,budget='3+2',slides=len(plans),tiles=sum(len(x['tiles']) for x in plans),prior_root=str(prior),prior_acceptance_sha256=sha(prior/'acceptance_v2.json'),status='prepared_geometry_and_coordinate_accepted'))
if __name__=='__main__':main()
