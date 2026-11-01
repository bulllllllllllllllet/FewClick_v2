"""Three-view bridge inputs; query labels are a separate terminal supervision path."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from benchmarks.v4.phase_4_cross_scale.src.export_tokens import he_to_tensor
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha

BASE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')


class BridgeData:
    def __init__(self):
        _,self.rows=load_cell_catalog(TASKS)
        self.lemon=BASE/'bridge_lemon_full_1101_133430/data'
        proof=json.loads((self.lemon/'independent_acceptance.json').read_text())
        assert proof['status']=='accepted_multiscale_LEMON_full' and proof['validation_fitting_cells']==0
        assert sha(self.lemon/'index.jsonl')==proof['index_sha256']
        self.entries={r['tile_id']:r for r in map(json.loads,(self.lemon/'index.jsonl').read_text().splitlines())}
        assert self.entries.keys()==self.rows.keys()
        transforms=BASE/'continuous_cell_grids_layout_repair_1031_145335/full/transforms.npz'
        assert sha(transforms)==proof['identity']['transforms_sha256']
        with np.load(transforms,allow_pickle=False) as f:self.mean=f['xcell_mean'].copy();self.std=f['xcell_std'].copy()
        self.identity=dict(lemon_index_sha256=proof['index_sha256'],transforms_sha256=sha(transforms))

    @lru_cache(maxsize=24)
    def inputs(self,identity,cell_arm):
        assert cell_arm in ['xcell','lemon'];row=self.rows[identity];variant=json.loads(row['variants_json'])['original']
        assert sha(variant['image'])==variant['image_sha256'] and sha(variant['cells'])==variant['cells_sha256']
        with Image.open(variant['image']) as im:images=[np.asarray(im.convert('RGB')).copy()]
        with np.load(variant['cells'],allow_pickle=False) as f:cells=[f['features'].copy()];totals=[int(f['total'])]
        assert sha(row['multiscale_cache'])==row['multiscale_sha256']
        with np.load(row['multiscale_cache'],allow_pickle=False) as f:
            for s in [1,2]:images.append(f['image_'+str(s)].copy());cells.append(f['cells_'+str(s)].copy());totals.append(int(f['total_'+str(s)]))
        entry=self.entries[identity];assert sha(entry['file'])==entry['file_sha256']
        result={}
        with np.load(entry['file'],allow_pickle=False) as f:
            for scale in range(3):
                old=cells[scale];assert old.ndim==2 and old.shape[1]==74
                if cell_arm=='lemon':value=f['cells_'+str(scale)].copy()
                else:value=np.concatenate([old[:,:10],(old[:,10:].astype(np.float64)-self.mean)/self.std],1).astype(np.float32)
                assert value.shape==old.shape and np.array_equal(value[:,:10],old[:,:10]) and np.isfinite(value).all()
                suffix='' if scale==0 else '_'+str(scale)
                result['image'+suffix]=he_to_tensor(images[scale])[None]
                result['cells'+suffix]=torch.from_numpy(value)[None]
                result['cell_valid'+suffix]=torch.ones(1,len(value),dtype=torch.bool)
                result['total_cell_count' if scale==0 else 'total_'+str(scale)]=torch.tensor([totals[scale]],dtype=torch.float32)
        return result

    def model_inputs(self,plan,cell_arm,device):
        move=lambda identity:{k:v.to(device) for k,v in self.inputs(identity,cell_arm).items()}
        prompts={}
        for sign in ['positive','negative']:
            xy=np.asarray(plan[sign+'_xy']);assert xy.ndim==2 and xy.shape[1]==2 and len(xy) and ((xy>0)&(xy<1)).all()
            prompts[sign+'_xy']=torch.tensor([plan[sign+'_xy']],dtype=torch.float32,device=device)
            prompts[sign+'_mask']=torch.ones(1,len(xy),dtype=torch.bool,device=device)
        prompts['prompt_size_id']=torch.ones(1,dtype=torch.long,device=device)
        local=plan['support_id']==plan['query_id']
        support=move(plan['support_id']);query=support if local else move(plan['query_id'])
        return dict(support=support,query=query,prompts=prompts,local=local,shared=local)

    @lru_cache(maxsize=32)
    def labels(self,identity):
        row=self.rows[identity]
        with Image.open(row['labels']) as im:truth=np.asarray(im).copy()
        import hashlib
        assert hashlib.sha256(truth.tobytes()).hexdigest()==row['labels_array_sha256']
        return truth,np.asarray(json.loads(row['input_mpp_xy_json']))
