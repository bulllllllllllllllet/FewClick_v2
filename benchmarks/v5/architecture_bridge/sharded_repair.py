"""Explicit manifest shard wrapper around the unchanged, accepted cell extractor."""
import argparse
import json
import sys
from pathlib import Path
from . import repair_cell_cache as worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shard',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gate-receipt',type=Path,required=True);p.add_argument('--accept',action='store_true')
    a=p.parse_args();contract=json.loads(a.manifest.read_text());wanted=contract['shards'][a.shard]
    original=worker.load_cell_catalog
    def subset(tasks):
        catalog,rows=original(tasks);assert set(wanted)<=rows.keys()
        return catalog,{k:rows[k] for k in wanted}
    worker.load_cell_catalog=subset
    sys.argv=[sys.argv[0],'--output',str(a.output),'--gate-receipt',str(a.gate_receipt)]+(['--accept'] if a.accept else [])
    worker.main()


if __name__=='__main__':main()
