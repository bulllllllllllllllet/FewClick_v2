"""Close bounded P1 screening only after prior audit gaps have concrete evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
    initial_path=base/'p1_obligation_audit_1101_122702/data/audit.json';initial=json.loads(initial_path.read_text())
    assert initial['status']=='complete_P1_obligation_audit_with_gaps'
    def verify(item):
        if isinstance(item,dict):
            if 'path' in item and 'sha256' in item:assert sha(item['path'])==item['sha256']
            for value in item.values():verify(value)
        elif isinstance(item,list):
            for value in item:verify(value)
    verify(initial['evidence'])
    selection=json.loads(a.selection.read_text());assert selection['image']=='frozen_phikon' and selection['cell_research_candidate']=='lemon_10um_center'
    assert selection['tissue_replacement_proven'] is False and selection['fullWSI_promoted'] is False
    assert selection['native_XCell_control_retained'] and selection['P2_requires_new_preregistered_contract']
    for path,digest in selection['bindings'].items():assert sha(path)==digest
    cost_path=base/'native_xcell_cost_1101_122838/data/report.json';cost=json.loads(cost_path.read_text())
    assert cost['status']=='accepted_native_XCell_component_cost' and len(cost['records'])==4
    assert sum(r['retained'] for r in cost['records'])==654
    for r in cost['records']:
        assert r['original_feature_max_abs_error']<=1e-5 and len(r['warm_repeats'])==3
        for t in r['warm_repeats']:
            assert all(t[k]>0 for k in ['crop_seconds','ctranspath_seconds','xcell_seconds'])
    grids=base/'continuous_cell_grids_layout_repair_1031_145335/full';proof=json.loads((grids/'independent_acceptance.json').read_text())
    assert proof['validation_fitting_cells']==0 and proof['all_grid_arrays_exact']
    assert sha(grids/'transforms.npz')==proof['transforms_sha256'] and sha(grids/'index.jsonl')==proof['index_sha256']
    requirements=initial['requirements']
    for r in requirements:
        if r['id']=='P1C_native_XCell_cost_comparable_ledger':r.update(state='component_cost_accepted_with_scope',resolved_by=str(cost_path))
        if r['id']=='P1_candidate_freeze_and_transfer_selection':r.update(state='research_candidate_frozen_no_superiority_claim',resolved_by=str(a.selection))
    assert not any(r['state'] in ['missing','pending_synthesis'] for r in requirements)
    a.output.mkdir(exist_ok=False)
    result=dict(status='accepted_bounded_P1_encoder_screening',P1_complete=True,requirements=requirements,
                initial_audit_sha256=sha(initial_path),selection_sha256=sha(a.selection),selection=str(a.selection),
                native_cost_sha256=sha(cost_path),source_sha256=sha(__file__),P2_status='preparing_not_executed',
                quality_outcome='no proven tissue upgrade; residual and limited-joint negative results retained',
                future_obligations=['controlled P2 tissue mask comparison','P3 spatial/support transfer mechanism','P4 native SAM system baselines and WSI quality/cost','P5 independent confirmation and multi-seed','P6 reproducibility/paper package'],
                limitations=['within-cohort exposed spatial validation','two coarse BC/TC labels only','synthetic color robustness only','small-sample cost scopes differ; no equal-FLOPs or WSI latency claim'])
    (a.output/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'])


if __name__=='__main__':main()
