"""Same retained cells and metadata; LEMON center features for all three native views."""
import argparse,hashlib,json,os
from pathlib import Path
import pyvips
import numpy as np
from PIL import Image
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.encoder_research.cell_grids import cell_vectors
from benchmarks.v5.encoder_research.lemon_features import FrozenLemon
from benchmarks.v5.paper_multiscale.data import native_context

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
RAW=BASE/'continuous_cell_cache_import_repair_1031_144058/full'
GRIDS=BASE/'continuous_cell_grids_layout_repair_1031_145335/full'


def patches(rgb,positions,mpp):
    width,height=np.rint(10/np.asarray(mpp)).astype(int);assert min(width,height)>0
    result=[]
    for xy in positions:
        left,top=np.rint(xy*np.asarray(rgb.shape[1::-1])-[width/2,height/2]).astype(int)
        pad=((max(0,-top),max(0,top+height-rgb.shape[0])),(max(0,-left),max(0,left+width-rgb.shape[1])),(0,0))
        image=np.pad(rgb,pad,mode='edge') if any(v for pair in pad for v in pair) else rgb
        x,y=left+pad[1][0],top+pad[0][0]
        result.append(Image.fromarray(image[y:y+height,x:x+width]).resize((40,40),Image.Resampling.BICUBIC))
    return result,[float(width*mpp[0]),float(height*mpp[1])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',action='store_true');p.add_argument('--gate-acceptance',type=Path);p.add_argument('--accept',action='store_true');a=p.parse_args()
    _,inventory=load_cell_catalog(TASKS)
    selected=[min((r for r in inventory.values() if r['dataset']==d and r['split']==s),key=lambda r:r['tile_id']) for d in ['crc','tiger'] for s in ['train','val']] if a.gate else [inventory[k] for k in sorted(inventory)]
    fitting=json.loads((GRIDS/'independent_acceptance.json').read_text());assert fitting['validation_fitting_cells']==0
    assert sha(GRIDS/'transforms.npz')==fitting['transforms_sha256']
    raw_proof=json.loads((RAW/'independent_acceptance.json').read_text());assert raw_proof['index_sha256']==sha(RAW/'index.jsonl')
    raw_rows={r['tile_id']:r for r in map(json.loads,(RAW/'index.jsonl').read_text().splitlines())}
    with np.load(GRIDS/'transforms.npz',allow_pickle=False) as f:transform={k:f[k].copy(order='C') for k in f.files}
    identity=dict(source_sha256=sha(__file__),raw_index_sha256=sha(RAW/'index.jsonl'),transforms_sha256=fitting['transforms_sha256'])
    if a.accept:
        contract=json.loads((a.output/'contract.json').read_text());assert contract['identity']==identity and contract['gate']==a.gate
        rows=list(map(json.loads,(a.output/'index.jsonl').read_text().splitlines()));assert [r['tile_id'] for r in rows]==[r['tile_id'] for r in selected]
        total=[0,0,0]
        for row in rows:
            original=inventory[row['tile_id']];variant=json.loads(original['variants_json'])['original']
            assert sha(row['file'])==row['file_sha256'] and sha(original['multiscale_cache'])==original['multiscale_sha256']
            with np.load(variant['cells'],allow_pickle=False) as f:fine=f['features'].copy()
            with np.load(original['multiscale_cache'],allow_pickle=False) as f:old=[fine,f['cells_1'].copy(),f['cells_2'].copy()]
            with np.load(row['file'],allow_pickle=False) as f:
                for scale,cells in enumerate(old):
                    value=f['cells_'+str(scale)];raw=f['lemon_'+str(scale)]
                    assert value.shape==cells.shape and np.array_equal(value[:,:10],cells[:,:10])
                    expected=cell_vectors(dict(metadata10=cells[:,:10],lemon384=raw),transform,'lemon')
                    assert np.array_equal(value,expected) and np.isfinite(value).all();total[scale]+=len(value)
        report=dict(status='accepted_multiscale_LEMON_gate' if a.gate else 'accepted_multiscale_LEMON_full',identity=identity,frames=len(rows),retained_cells_by_scale=total,worker_pid=os.getpid(),index_sha256=sha(a.output/'index.jsonl'),validation_fitting_cells=0,scope='same retained row-order/metadata; context masks unavailable; no detector or tissue-label use')
        (a.output/'independent_acceptance.json').write_text(json.dumps(report,indent=2)+'\n');return
    if not a.gate:
        assert a.gate_acceptance;proof=json.loads(a.gate_acceptance.read_text());assert proof['status']=='accepted_multiscale_LEMON_gate' and proof['identity']==identity
    a.output.mkdir(exist_ok=False)
    (a.output/'contract.json').write_text(json.dumps(dict(identity=identity,gate=a.gate,context_RGB='native_context at audited full footprint; crop10um then40px; no512-downsample substitute',fine_RGB='reuse accepted original P1 fine crop/LEMON cache'),indent=2))
    encoder=FrozenLemon();reload_test=None
    with (a.output/'index.jsonl').open('w') as stream:
        for row in selected:
            variant=json.loads(row['variants_json'])['original'];entry=raw_rows[row['tile_id']]
            assert sha(entry['file'])==entry['file_sha256'] and sha(variant['cells'])==variant['cells_sha256']
            assert sha(row['multiscale_cache'])==row['multiscale_sha256']
            with np.load(variant['cells'],allow_pickle=False) as f:fine=f['features'].copy()
            with np.load(entry['file'],allow_pickle=False) as f:values=[f['lemon384'].copy()]
            with np.load(row['multiscale_cache'],allow_pickle=False) as f:old=[fine,f['cells_1'].copy(),f['cells_2'].copy()]
            arrays={};geometry=json.loads(row['multiscale_json'])
            for scale in [1,2]:
                crops,actual=patches(native_context(row,scale),old[scale][:,:2],geometry['native_mpp'])
                features=np.concatenate([encoder.encode(crops[i:i+32]) for i in range(0,len(crops),32)]) if crops else np.empty((0,384),np.float32)
                values.append(features)
                if reload_test is None and crops:reload_test=(crops[:4],features[:4].copy())
                arrays['actual_crop_um_'+str(scale)]=np.asarray(actual)
            for scale in range(3):
                assert len(values[scale])==len(old[scale])
                arrays['lemon_'+str(scale)]=values[scale]
                arrays['cells_'+str(scale)]=cell_vectors(dict(metadata10=old[scale][:,:10],lemon384=values[scale]),transform,'lemon')
            path=a.output/(row['tile_id']+'.npz');np.savez_compressed(path,**arrays)
            stream.write(json.dumps(dict(tile_id=row['tile_id'],file=str(path),file_sha256=sha(path)))+'\n');stream.flush()
            print(json.dumps(dict(tile_id=row['tile_id'],completed=True)),flush=True)
    if a.gate:
        del encoder;encoder=FrozenLemon();crops,expected=reload_test
        assert np.allclose(encoder.encode(crops),expected,atol=1e-5,rtol=1e-5)
    (a.output/'build_complete.json').write_text(json.dumps(dict(worker_pid=os.getpid(),frames=len(selected),fresh_encoder_reload_passed=a.gate))+'\n')


if __name__=='__main__':main()
