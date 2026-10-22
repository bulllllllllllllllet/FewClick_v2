"""Build rsync --files-from lists from the actual checkpoint/data manifests.

Run on the source server (or a server already sharing its NFS). Paths in the
list are relative to filesystem root; rsync's source must therefore end in /.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['bootstrap','probe','validation','training','resume','preprocess'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=Path(__file__).resolve().parent
    manifest=json.loads((base/'assets.json').read_text());catalog=Path(manifest['catalog'])
    categories={'bootstrap','trained','training_metadata','catalog'}
    if args.scope=='resume':categories.add('resume4k')
    if args.scope=='preprocess':categories.add('new_preprocessing')
    files={e['path'] for e in manifest['entries'] if e['external'] and e['category'] in categories}
    if args.scope in ('probe','validation','training','resume'):
        frame=pd.read_parquet(catalog/'tiles.parquet');rows={r['tile_id']:r for r in frame.to_dict('records')}
        if args.scope in ('training','resume'):
            selected={(tile,q) for tile in rows for q in ('original','0.8','1.0')}
        else:
            path=base/'probe_plans.json' if args.scope=='probe' else Path(manifest['training_run'])/'validation_plans.json'
            plans=json.loads(path.read_text());selected={(p[role+'_id'],p[role+'_quality']) for p in plans for role in ('support','query')}
        for tile,quality in selected:
            row=rows[tile];variant=json.loads(row['variants_json'])[quality]
            files.update([row['labels'],variant['image'],variant['cells']])
    for name in files:
        if not Path(name).is_absolute() or '\n' in name:raise ValueError('invalid absolute source path: '+repr(name))
    data=('\n'.join(sorted(name.lstrip('/') for name in files))+'\n').encode()
    # Never replace an existing migration list.
    with args.output.open('xb') as stream:stream.write(data)
    print(json.dumps(dict(scope=args.scope,files=len(files),list_sha256=hashlib.sha256(data).hexdigest(),
        output=str(args.output),rsync_source='SOURCE_HOST:/',relative_to='filesystem root',
        note='List includes actual referenced files, not whole cache directories; bootstrap does not include RGB/cell/GT inputs.')))


if __name__=='__main__':main()
