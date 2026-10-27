"""Supplement decorated-function provenance with historical original input recipes."""
import argparse
import inspect
import json
from pathlib import Path
import os
import subprocess

from .audit_heldout_local import read,sha,require
from benchmarks.v5.multidomain_data.prepare_cells import prepare_variant


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    actual=Path(inspect.getfile(inspect.unwrap(prepare_variant))).resolve()
    wrapped=Path(inspect.getfile(prepare_variant)).resolve()
    require(actual.name=='prepare_cells.py','actual feature preparation source')
    historical=Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/instanseg_adoption_1020_185343/full_cells_1020_205325/shard_02/run.json')
    original=read(historical)
    require(original['recipe']['source_sha256'][str(actual)]==sha(str(actual)),'historical/current unwrapped function source identity')
    require(not subprocess.check_output(['git','diff','--',str(actual)],text=True),'feature preparation worktree unchanged')
    meta=read(a.root/'pca_full_train_1025_234700/run.json');matches=[]
    for folder in map(Path,meta['raw_shards']):
        runmeta=read(folder/'run.json')
        require(runmeta['recipe']==meta['raw_recipe'],'same raw recipe')
        require(runmeta['recipe']['sources'][str(wrapped)]==sha(str(wrapped)),'historically hashed decorator wrapper')
        matches.append(dict(folder=str(folder),run_sha256=sha(str(folder/'run.json'))))
    result=dict(status='passed_decorated_source_provenance_supplement',worker_pid=os.getpid(),
        actual_source=str(actual),actual_source_sha256=sha(str(actual)),decorator_source=str(wrapped),decorator_source_sha256=sha(str(wrapped)),
        historical_input_recipe=str(historical),historical_input_recipe_sha256=sha(str(historical)),raw_groups=matches,
        last_tracked_change=subprocess.check_output(['git','log','-1','--format=%H %ai','--',str(actual)],text=True).strip(),source_sha256=sha(__file__),
        limits=['Raw metadata hashed decorated Torch wrapper rather than directly the unwrapped source; original records preserved',
                'This records historical input recipe/current tracked-source identity; it cannot retrospectively prove absence of transient uncommitted runtime changes',
                'Full cache identity and raw/reference74 parity independently audited; no new feature extraction or training'])
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
