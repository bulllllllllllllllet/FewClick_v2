"""Independent all-file cache identity check, without loading an encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .image_features import CHANNELS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.cache/'report.json').read_text());rows=[json.loads(s) for s in (a.cache/'index.jsonl').read_text().splitlines()]
    catalog=pd.read_parquet('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')
    expected={x['tile_id']:json.loads(x['variants_json'])['original']['image_sha256'] for x in catalog.to_dict('records')}
    assert len(rows)==len(expected)==1737 and {x['tile_id'] for x in rows}==set(expected)
    for row in rows:
        assert row['image_sha256']==expected[row['tile_id']]
        feature=np.load(row['feature'],allow_pickle=False)
        assert feature.shape==(CHANNELS[r['arm']],32,32) and feature.dtype==np.float16 and np.isfinite(feature).all()
        assert hashlib.sha256(feature.tobytes()).hexdigest()==row['array_sha256']
    assert hashlib.sha256((a.cache/'index.jsonl').read_bytes()).hexdigest()==r['index_sha256']
    result=dict(status='accepted_all1737_spatial_features',arm=r['arm'],rows=len(rows),index_sha256=r['index_sha256'],finite_and_array_sha_rechecked=True)
    (a.cache/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
