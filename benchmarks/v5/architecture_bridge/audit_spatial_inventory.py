"""All-scale sampling coverage and pixel-exact view audit, without tissue labels."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy import ndimage
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--gate-receipt',type=Path);a=p.parse_args()
    proof=json.loads((a.cache/'acceptance.json').read_text());assert proof['status']=='accepted_repaired_normalized_cells'
    assert sha(a.cache/'index.jsonl')==proof['index_sha256']
    entries={r['tile_id']:r for r in map(json.loads,(a.cache/'index.jsonl').read_text().splitlines())}
    _,inventory=load_cell_catalog(TASKS);assert entries.keys()==inventory.keys()
    selected=[min(k for k,v in inventory.items() if v['dataset']==d and v['split']==s) for d in ['crc','tiger'] for s in ['train','val']] if a.gate else sorted(entries)
    identity=dict(source_sha256=sha(__file__),index_sha256=proof['index_sha256'])
    if not a.gate:
        gate=json.loads(a.gate_receipt.read_text());assert gate['status']=='accepted_spatial_inventory_gate' and gate['identity']==identity
    a.output.mkdir(exist_ok=False);records=[];started=time.monotonic()
    with (a.output/'frames.jsonl').open('w') as stream:
        for tile in selected:
            row=inventory[tile];entry=entries[tile];assert sha(entry['file'])==entry['sha256']
            variant=json.loads(row['variants_json'])['original'];assert sha(variant['image'])==variant['image_sha256'] and sha(row['multiscale_cache'])==row['multiscale_sha256']
            with Image.open(variant['image']) as im:fine=np.asarray(im.convert('RGB')).copy()
            with np.load(row['multiscale_cache']) as old,np.load(entry['file']) as raw:
                for s in range(3):
                    reference=fine if s==0 else old[f'image_{s}']
                    assert np.array_equal(reference,raw[f'image_{s}']), (tile,s,'image_view_mismatch')
                    inst,ids,meta=raw[f'instances_{s}'],raw[f'ids_{s}'],raw[f'meta_{s}']
                    allids=np.unique(inst);allids=allids[allids>0];n=len(allids);assert n==int(raw[f'total_{s}']) and len(ids)==min(n,255)
                    yx=np.asarray(ndimage.center_of_mass(np.ones(inst.shape,np.uint8),inst,allids)).reshape(-1,2) if n else np.empty((0,2))
                    xy=(yx[:,::-1]+.5)/np.array(inst.shape[::-1]);q=(xy[:,1]>=.5).astype(int)*2+(xy[:,0]>=.5).astype(int)
                    full=np.bincount(q,minlength=4);chosen=np.bincount(q[np.isin(allids,ids)],minlength=4)
                    tv=float(np.abs(full/n-chosen/len(ids)).sum()/2) if n else None
                    physical=np.array(inst.shape[::-1])*raw[f'mpp_{s}']
                    expected=np.array(json.loads(row['input_mpp_xy_json']))*512 if s==0 else np.array(json.loads(row['multiscale_json'])['native_boxes'][s][2:])*np.array(json.loads(row['multiscale_json'])['native_mpp'])
                    assert np.allclose(physical,expected,atol=1e-6)
                    item=dict(tile_id=tile,dataset=row['dataset'],split=row['split'],scale=s,detected=n,retained=len(ids),capped=n>255,empty=n==0,
                              full_quadrants=full.tolist(),retained_quadrants=chosen.tolist(),quadrant_TV=tv,
                              physical_extent_um=physical.tolist(),source_file_sha256=entry['sha256'])
                    records.append(item);stream.write(json.dumps(item)+'\n')
            stream.flush()
            status=dict(status='running',worker_pid=os.getpid(),completed=len(records)//3,total=len(selected),seconds=time.monotonic()-started)
            (a.output/'status.json').write_text(json.dumps(status))
            if len(records)%150==0:print(json.dumps(status),flush=True)
    # Independently re-read the exported rows to reconstruct all denominators.
    saved=list(map(json.loads,(a.output/'frames.jsonl').read_text().splitlines()));assert saved==records
    assert len(saved)==3*len(selected) and len({(r['tile_id'],r['scale']) for r in saved})==len(saved)
    groups=defaultdict(list)
    for row in saved:
        assert sum(row['full_quadrants'])==row['detected'] and sum(row['retained_quadrants'])==row['retained']
        groups[f"{row['dataset']}|{row['split']}|{row['scale']}"].append(row)
    summary={}
    for key,rows in groups.items():
        tv=[r['quadrant_TV'] for r in rows if r['capped']]
        summary[key]=dict(fields=len(rows),detected=sum(r['detected'] for r in rows),retained=sum(r['retained'] for r in rows),
                          capped_fields=sum(r['capped'] for r in rows),empty_fields=sum(r['empty'] for r in rows),
                          capped_TV_mean=float(np.mean(tv)) if tv else None,capped_TV_max=max(tv) if tv else None)
    receipt=dict(status='accepted_spatial_inventory_gate' if a.gate else 'accepted_full_spatial_inventory',identity=identity,fields=len(selected),views=len(records),
                 groups=summary,frames_sha256=sha(a.output/'frames.jsonl'),GT_used=False,all_image_views_pixel_exact=True,physical_extents_checked=True,
                 limitation='quadrant coverage describes retained detection samples, not celltype coverage or segmentation quality')
    (a.output/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.output/'status.json').write_text(json.dumps(dict(status='complete',worker_pid=os.getpid(),fields=len(selected))))


if __name__=='__main__':main()
