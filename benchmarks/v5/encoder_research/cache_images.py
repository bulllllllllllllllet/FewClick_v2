"""Bounded frozen spatial cache, image-only, immutable identity and precision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pandas as pd
import torch
from .image_features import FrozenImageEncoder,CHANNELS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=list(CHANNELS),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.time();torch.set_num_threads(2)
    catalog=Path('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')
    rows=pd.read_parquet(catalog).sort_values('tile_id').to_dict('records');assert len(rows)==1737
    status=dict(status='loading',worker_pid=os.getpid(),arm=a.arm,completed=0,total=len(rows))
    def save():
        status['elapsed_seconds']=time.time()-start
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();encoder=FrozenImageEncoder(a.arm);index=[]
    with (a.output/'index.jsonl').open('w') as stream:
        for n,row in enumerate(rows):
            variant=json.loads(row['variants_json'])['original'];assert sha(variant['image'])==variant['image_sha256']
            with Image.open(variant['image']) as im:rgb=np.asarray(im.convert('RGB')).copy()
            assert rgb.shape==(512,512,3)
            feature=encoder(torch.from_numpy(rgb).permute(2,0,1)[None]).cpu().numpy()[0]
            stored=feature.astype('float16');assert np.isfinite(stored).all()
            relative=float(np.linalg.norm(stored.astype('float32')-feature)/max(np.linalg.norm(feature),1e-12))
            assert relative<.001
            path=a.output/(row['tile_id']+'.npy');temporary=path.with_suffix('.partial')
            with temporary.open('wb') as f:np.save(f,stored,allow_pickle=False)
            temporary.rename(path)
            record=dict(tile_id=row['tile_id'],split=row['split'],dataset=row['dataset'],image=variant['image'],image_sha256=variant['image_sha256'],
                feature=str(path),array_sha256=hashlib.sha256(stored.tobytes()).hexdigest(),shape=list(stored.shape),dtype=str(stored.dtype),
                precision_relative_L2=relative,bytes=path.stat().st_size)
            stream.write(json.dumps(record)+'\n');stream.flush();index.append(record)
            if n==0:(a.output/'input_precision_gate.json').write_text(json.dumps(dict(status='passed_real_input_cache_gate',row=record))+'\n')
            status.update(status='running',completed=n+1);save()
    report=dict(status='complete_image_cache_unaccepted',worker_pid=os.getpid(),arm=a.arm,rows=len(index),
        source_sha256=sha(__file__),catalog_sha256=sha(catalog),index_sha256=sha(a.output/'index.jsonl'),
        total_bytes=sum(x['bytes'] for x in index),max_precision_relative_L2=max(x['precision_relative_L2'] for x in index),
        elapsed_seconds=time.time()-start,GT_read=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');status.update(status='complete');save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
