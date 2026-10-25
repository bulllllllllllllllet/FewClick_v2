"""Same frozen remote mechanics with explicitly all-label G3 reference weights."""
import argparse
from pathlib import Path

from benchmarks.v5.heldout_transfer import evaluate as engine
from benchmarks.v5.heldout_comparison.audit import FOLDS
from benchmarks.v5.fair_comparison.audit import digest

BASE_SOURCES=engine.sources
BASE_WRITE=engine.write
REFERENCE_SCOPE='All-label G3 training AND all-target local development checkpoint selection; identical heldout task reference, not heldout-trained accuracy or isolated supervision effect'


def reference_sources():return BASE_SOURCES()|{str(Path(__file__)):digest(__file__)}


def reference_write(path,value):
    if not any(key in value for key in ('status','method','adapted_method')):
        BASE_WRITE(path,value)
        return
    BASE_WRITE(path,value|dict(all_label_reference=True,scope=REFERENCE_SCOPE,
        selection_information='Original full302-pair development set includes the scored target; differs from excluded-label seen-only selection'))


def validate_reference(a):
    read=engine.read;done=read(a.training/'summary.json');run=read(a.training/'run.json');best=read(a.training/'best_checkpoint.json')
    expected='fe8311f9acf29c500e93ceb89f1f466d0241d680598bcc6f9ae8ee49adf91afe'
    if done['status']!='complete' or done['step']!=10000 or not done['full_manifest_parity'] or done['training_consumed_sha256']!=expected:
        raise ValueError('accepted full-label G3 adaptation required')
    accepted=read(a.development.parent/'G3_complete_audit_1025_155140.json')
    if accepted['status']!='passed' or best!=accepted['methods']['wsi_sam_ft']['selected_checkpoint']:raise ValueError('accepted full-label selected weight changed')
    if digest(best['path'])!=best['checkpoint_sha256'] or digest(a.training/'training_consumed.jsonl')!=expected or digest(a.realized/'training_plans.jsonl')!=expected:
        raise ValueError('actual all-label training/weight identity mismatch')
    if run['contract']['method']!='wsi_sam_ft' or run['contract']['protocol_sha256']!=digest(a.development/'contract.json'):
        raise ValueError('G3 checkpoint source protocol mismatch')
    test=read(a.tasks/'contract.json');fold=test['heldout_fold'];plans=read(a.tasks/'validation_maximum.json')
    if fold not in FOLDS or test['checkpoint_selection_allowed'] is not False or digest(a.tasks/'validation_maximum.json')!=test['tasks_sha256']:
        raise ValueError('frozen heldout target subset required')
    if any(p['target'] not in FOLDS[fold]['test'][p['dataset']] for p in plans):raise ValueError('test subset membership')
    if test['catalog_sha256']!=read(a.development/'contract.json')['catalog_sha256']:raise ValueError('same original test image/GT catalog required')
    return dict(fold=fold,checkpoint=best['path'],checkpoint_sha256=best['checkpoint_sha256'],
        training_protocol_sha256=digest(a.development/'contract.json'),selected_seen_step=best['step'],training_contract=run['contract']),expected,plans


def main():
    p=argparse.ArgumentParser()
    for name in ('tasks','development','training','realized','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--verified-gate',type=Path);a=p.parse_args()
    a.method='wsi_sam_ft';a.checkpoint_gate=False
    engine.validate=validate_reference;engine.sources=reference_sources;engine.write=reference_write
    engine.run(a)


if __name__=='__main__':main()
