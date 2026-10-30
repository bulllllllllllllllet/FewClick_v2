"""Freeze organ/class balanced cell probes with bounded native input views."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def key(s):return hashlib.sha256(s.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inventory=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/p0_ocelot_cells_1030_182153/inventory.json')
    r=json.loads(inventory.read_text());assert not r['cross_official_split_full_input_overlaps']
    sources={x['id']:x for x in r['sources']};buckets=defaultdict(lambda:defaultdict(list));excluded=Counter()
    for x in r['non_test_cell_inputs']:
        points=np.loadtxt(x['annotation'],delimiter=',',ndmin=2) if Path(x['annotation']).read_text().strip() else np.empty((0,3))
        source=sources[x['id']];mx,my=source['cell_mpp_xy']
        for index,(cx,cy,label) in enumerate(points):
            w,h=round(40/mx),round(40/my);left,top=round(cx-w/2),round(cy-h/2)
            if min(left,top)<0 or left+w>1024 or top+h>1024:
                excluded[f"{x['split']}|{x['organ']}|{int(label)}"]+=1;continue
            row=dict(cell_id=f"ocelot:{x['id']}:{index}",point_index=index,field_id=x['id'],
                split=x['split'],organ=x['organ'],label=int(label),xy=[cx,cy],image=x['image'],
                image_sha256=x['image_sha256'],annotation=x['annotation'],annotation_sha256=x['annotation_sha256'],
                mpp_xy=[mx,my],slide=source['slide'],patient=source['patient'],
                full_available_input_box_level0=source['full_available_input_box_level0'],
                max_crop_pixel_box=[left,top,w,h],instance_mask_available=False)
            buckets[(x['split'],x['organ'],int(label))][x['id']].append(row)
    selected=[];coverage=[]
    for split in ['train','val']:
        for organ in sorted(r['selected_train_ROIs']):
            for label in [1,2]:
                fields=buckets[(split,organ,label)];cap=32 if split=='train' else 16
                for values in fields.values():values.sort(key=lambda x:key(x['cell_id']))
                ordered=sorted(fields,key=key);rows=[]
                # At most4 points/field/class; round-robin broadens spatial source coverage.
                for index in range(4):
                    for identity in ordered:
                        if len(fields[identity])>index and len(rows)<cap:rows.append(fields[identity][index])
                selected.extend(rows);coverage.append(dict(split=split,organ=organ,label=label,
                    selected=len(rows),requested=cap,eligible=sum(map(len,fields.values())),fields=len(fields)))
    with (a.output/'cells.jsonl').open('w') as f:
        for row in selected:f.write(json.dumps(row)+'\n')
    # Execute the largest crop and common10um canonical view on one cell per stratum.
    gates=[]
    for group in coverage:
        matching=[x for x in selected if all(x[k]==group[k] for k in ['split','organ','label'])]
        if not matching:continue
        x=matching[0];mx,my=x['mpp_xy'];left,top,w,h=x['max_crop_pixel_box']
        with Image.open(x['image']) as im:
            rgb=im.convert('RGB');patch=np.asarray(rgb.crop((left,top,left+w,top+h)));assert patch.shape==(h,w,3)
            cx,cy=x['xy'];cw,ch=round(10/mx),round(10/my);xx,yy=round(cx-cw/2),round(cy-ch/2)
            target=a.output/(x['cell_id'].replace(':','_')+'.png')
            rgb.crop((xx,yy,xx+cw,yy+ch)).resize((40,40),Image.Resampling.BICUBIC).save(target)
        gates.append(dict(cell_id=x['cell_id'],crop=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    report=dict(status='complete_probe_manifest_not_encoder_training',rows=len(selected),coverage=coverage,
        excluded_max40um_edge_counts=dict(excluded),gates=gates,
        inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        cells_sha256=hashlib.sha256((a.output/'cells.jsonl').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract=dict(task='expertBC/TC classification and train-only retrieval diagnostic, not tissue segmentation',
            fit_split='train',selection_split='val',confirmation='not yet frozen, test labels unread',
            cap_per_organ_class=dict(train=32,val=16),max_points_per_field_class=4,
            selection='identity SHA round robin across fields; full40um inbounds required; no model scores',
            allowed_crop_extents_um=[10,32,40],common_view_um=10,
            native_view_conditions='32um XCell attended recipe requires detector masks;40um pathology center tokens separate from common10um; never call unequal views matched information',
            XCell_point_mask_policy='No expert instance masks; detector matching contract still required for unchanged attended-crop baseline',
            zero_fields='all5 retained in inventory; zero-cell cannot produce cell classification rows',
            exposure='historically used OCELOT/TCGA; dev only, no fresh blind-test assertion'))
    (a.output/'contract.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(rows=len(selected),gates=len(gates),coverage=coverage)))


if __name__=='__main__':main()
