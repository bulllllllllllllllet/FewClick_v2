"""Train-only k8 physical scale audit; no tissue GT or neighborhood training."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import numpy as np
from .geometry import physical_knn
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha


def raster_shape(npz,key):
    with npz.zip.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError(f'Unsupported NPY format {version}')
    assert len(shape)==2 and np.issubdtype(dtype,np.integer);return shape


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--gate-receipt',type=Path);a=p.parse_args()
    proof=json.loads((a.cache/'acceptance.json').read_text());assert proof['status']=='accepted_repaired_normalized_cells' and sha(a.cache/'index.jsonl')==proof['index_sha256']
    entries={r['tile_id']:r for r in map(json.loads,(a.cache/'index.jsonl').read_text().splitlines())};_,catalog=load_cell_catalog(TASKS)
    selected=sorted(k for k,row in catalog.items() if row['split']=='train');assert len(selected)==1356
    if a.gate:selected=sum([sorted(k for k in selected if catalog[k]['dataset']==d)[:2] for d in ['crc','tiger']],[])
    identity=dict(source_sha256=sha(__file__),geometry_sha256=sha(Path(__file__).with_name('geometry.py')),input_index_sha256=proof['index_sha256'],k=8,units='um')
    if not a.gate:
        prior=json.loads(a.gate_receipt.read_text());assert prior['status']=='accepted_train_knn_geometry_gate' and prior['identity']==identity
    a.output.mkdir(exist_ok=False);groups=defaultdict(list);records=[];started=time.monotonic()
    for number,tile in enumerate(selected):
        entry=entries[tile];row=catalog[tile];assert entry['split']=='train' and sha(entry['file'])==entry['sha256']
        with np.load(entry['file'],allow_pickle=False) as raw:
            for scale in range(3):
                ids=raw[f'ids_{scale}'];meta=raw[f'meta_{scale}'];shape=raster_shape(raw,f'instances_{scale}');extent=np.array(shape[::-1])*raw[f'mpp_{scale}']
                xy=meta[:,:2].astype(np.float64)*extent;neighbors,distances=physical_knn(xy,ids)
                assert len(ids)<=255 and ((meta[:,:2]>0)&(meta[:,:2]<1)).all()
                # Independent scalar-distance sorting checks, including tie-breaking.
                probes=range(len(ids)) if a.gate else sorted({0,len(ids)//2,len(ids)-1}) if len(ids) else []
                for i in probes:
                    ordered=sorted((float(np.hypot(*(xy[i]-xy[j]))),int(ids[j]),j) for j in range(len(ids)) if i!=j)[:8]
                    assert neighbors[i].tolist()==[x[2] for x in ordered]
                    assert np.allclose(distances[i],[x[0] for x in ordered],atol=1e-10)
                name=f"{row['dataset']}|{scale}";groups[name].append(distances.ravel())
                records.append(dict(tile_id=tile,dataset=row['dataset'],scale=scale,retained=len(ids),detected=int(raw[f'total_{scale}']),
                    edges=distances.size,extent_um=extent.tolist(),median_edge_um=float(np.median(distances)) if distances.size else None,
                    max_edge_um=float(distances.max()) if distances.size else None,source_file_sha256=entry['sha256']))
        if number%25==0:
            status=dict(status='running',completed=number+1,total=len(selected),worker_pid=os.getpid(),seconds=time.monotonic()-started)
            (a.output/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
    summary={}
    for key,parts in groups.items():
        values=np.concatenate(parts);summary[key]=dict(edges=len(values),quantile50_90_95_um=np.quantile(values,[.5,.9,.95]).tolist() if len(values) else None,
            fraction_longer_than_um={str(t):float(np.mean(values>t)) if len(values) else None for t in [32,64,128,256]})
    (a.output/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    report=dict(status='accepted_train_knn_geometry_gate' if a.gate else 'accepted_train_knn_geometry_audit',identity=identity,train_frames=len(selected),views=len(records),summary=summary,
        frames_sha256=sha(a.output/'frames.jsonl'),val_frames_read=0,tissue_GT_read=False,training_started=False,training_allowed=False,
        limitation='within-view retained-cell graph, not full detected/biological adjacency; views may repeat cells; no segmentation/WSI quality claim')
    (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'status.json').write_text(json.dumps(dict(status='complete',train_frames=len(selected))))


if __name__=='__main__':main()
