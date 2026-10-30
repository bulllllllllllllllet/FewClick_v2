"""Audit same-raster expert cell points and source-space context footprints."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.time()
    data=Path('/nfs-medical3/zyh/OCELOT_1016_122128/extracted/ocelot2023_v1.0.1')
    meta=json.loads((data/'metadata.json').read_text());pairs=meta['sample_pairs'];sources=[];overlaps=[]
    for identity,m in sorted(pairs.items()):
        c,t=m['cell'],m['tissue'];b=[min(c['x_start'],t['x_start']),min(c['y_start'],t['y_start']),
            max(c['x_end'],t['x_end']),max(c['y_end'],t['y_end'])]
        sources.append(dict(id=identity,slide=m['slide_name'],patient=m['slide_name'][:12],
            split=m['subset'],organ=m['organ'],full_available_input_box_level0=b,
            cell_box_level0=[c[k] for k in ['x_start','y_start','x_end','y_end']],
            cell_mpp_xy=[c['resized_mpp_x'],c['resized_mpp_y']],
            provenance='official split retained, not newly patient-grouped'))
    for i,x in enumerate(sources):
        for y in sources[:i]:
            if x['slide']!=y['slide'] or x['split']==y['split']:continue
            b,c=x['full_available_input_box_level0'],y['full_available_input_box_level0']
            if min(b[2],c[2])>max(b[0],c[0]) and min(b[3],c[3])>max(b[1],c[1]):
                overlaps.append([x['id'],y['id']])
    inventory=[];small=[]
    # Select train diagnostic ROIs by identity hash, without consulting annotations.
    selected={}
    for organ in sorted({m['organ'] for m in pairs.values()}):
        ids=[i for i,m in pairs.items() if m['organ']==organ and m['subset']=='train']
        selected[organ]=min(ids,key=lambda v:hashlib.sha256(v.encode()).hexdigest())
    for s in sources:
        identity=s['id'];subset=s['split'];m=pairs[identity]
        if subset=='test':continue  # metadata-only inventory: never inspect test labels here.
        image=data/f'images/{subset}/cell/{identity}.jpg';labels=data/f'annotations/{subset}/cell/{identity}.csv'
        # Empty official annotations are valid zero-cell fields, retained in inventory.
        points=np.loadtxt(labels,delimiter=',',ndmin=2) if labels.read_text().strip() else np.empty((0,3))
        assert points.shape[1]==3 and np.isfinite(points).all()
        assert set(points[:,2]).issubset({1,2})
        with Image.open(image) as im:
            assert im.size==(1024,1024)
            assert ((points[:,:2]>=0)&(points[:,:2]<1024)).all()
            mx,my=s['cell_mpp_xy'];c=m['cell']
            assert np.allclose([mx,my],[(c['x_end']-c['x_start'])*m['mpp_x']/1024,
                                        (c['y_end']-c['y_start'])*m['mpp_y']/1024])
            inventory.append(dict(id=identity,split=subset,organ=s['organ'],points=len(points),
                counts=dict(Counter(points[:,2].astype(int).tolist())),image=str(image),image_sha256=sha(image),
                annotation=str(labels),annotation_sha256=sha(labels)))
            if selected[s['organ']]!=identity:continue
            rgb=im.convert('RGB')
            for label in (1,2):
                candidates=[]
                for index,(x,y,lab) in enumerate(points):
                    if lab!=label:continue
                    w,h=round(40/mx),round(40/my)
                    left,top=round(x-w/2),round(y-h/2)
                    if left>=0 and top>=0 and left+w<=1024 and top+h<=1024:
                        candidates.append((hashlib.sha256(f'{identity}|{index}'.encode()).hexdigest(),index,left,top,w,h))
                if not candidates:
                    small.append(dict(id=identity,label=label,status='no_40um_inbounds_point'));continue
                _,index,left,top,w,h=min(candidates);x,y,_=points[index]
                path=a.output/f'{identity}_{index}_context40um.png'
                rgb.crop((left,top,left+w,top+h)).save(path)
                cw,ch=round(10/mx),round(10/my);cx,cy=round(x-cw/2),round(y-ch/2)
                canonical=a.output/f'{identity}_{index}_cell10um_40px.png'
                rgb.crop((cx,cy,cx+cw,cy+ch)).resize((40,40),Image.Resampling.BICUBIC).save(canonical)
                small.append(dict(id=identity,point_index=index,label=label,organ=s['organ'],center_xy=[x,y],
                    native_mpp_xy=[mx,my],context_pixel_box=[left,top,w,h],actual_context_um=[w*mx,h*my],
                    actual_cell_um=[cw*mx,ch*my],context=str(path),canonical_cell=str(canonical),
                    canonical_sha256=sha(canonical),label_granularity='expert point, no instance mask',
                    status='decoded_diagnostic_only',training_allowed=False))
    report=dict(status='complete_ocelot_cell_input_inventory_not_training_contract',sources=sources,
        cross_official_split_full_input_overlaps=overlaps,non_test_cell_inputs=inventory,diagnostics=small,
        metadata_sha256=sha(data/'metadata.json'),source_sha256=sha(__file__),label_info=meta['label_info']['cell'],
        test_labels_read=False,selected_train_ROIs=selected,elapsed_seconds=time.time()-start,
        exposure='Existing OCELOT data had prior diagnostics. No assertion of previously unexposed confirmation.',
        limitations=['BC/TC only, not fine immune ontology','point annotation does not supply instance segmentation',
                     'foundation model TCGA pretraining exposure must be disclosed','new encoder run split/selection contract still pending'])
    (a.output/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],non_test_inputs=len(inventory),diagnostics=len(small),
                         cross_split_overlaps=len(overlaps),seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
