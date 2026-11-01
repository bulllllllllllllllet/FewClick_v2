"""Evidence inventory for original P1 obligations; never equates negative results with promotion."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    path=Path(path);data=json.loads(path.read_text())
    return data,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=data.get('status'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    base=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research');evidence={};cost={}
    for arm in ['resnet','phikon','sam']:
        cache,identity=read(base/'image_pilot_1031_015107'/arm/'cache/acceptance.json')
        report,proof=read(base/'image_pilot_1031_015107'/arm/'cache/report.json')
        assert cache['status']=='accepted_all1737_spatial_features' and report['rows']==1737
        assert cache['index_sha256']==report['index_sha256']
        index=base/'image_pilot_1031_015107'/arm/'cache/index.jsonl'
        assert hashlib.sha256(index.read_bytes()).hexdigest()==cache['index_sha256']
        cost[arm]=dict(rows=1737,cache_bytes=report['total_bytes'],elapsed_seconds=report['elapsed_seconds'],
                       scope='historical whole cache generation wall time; includes preprocessing/IO, not isolated GPU latency')
        evidence['image_'+arm]=dict(acceptance=identity,report=proof)
    for key,relative in {
        'image_comparison':'image_pilot_comparison_1031_022128/comparison.json',
        'joint_comparison':'phikon_joint_comparison_1101_122214/data/comparison.json',
        'cell_probe':'cell_linear_probe_1031_012400/independent_acceptance.json',
        'cell_uncertainty':'cell_probe_source_uncertainty_1031_151743/uncertainty.json',
        'cell_two_seed':'two_seed_cell_summary_1031_154415/result/summary.json',
        'cell_frozen_revision':'cell_only_analysis_1031_162255/result/comparison.json',
        'color_sensitivity':'cell_color_probe_1101_122516/data/report.json',
        'native_XCell_gate':'xcell_point_gate_1031_005709/independent_acceptance.json',
    }.items():
        data,proof=read(base/relative);evidence[key]=proof
        if key=='image_comparison':
            assert data['summaries'] and data['strata']; evidence[key]['strata_available']=True
        if key=='joint_comparison': assert data['eligible_for_second_seed'] is False and data['comparison']['tests']['primary_Dice_gain'] is False
        if key=='cell_frozen_revision': assert data['eligible_for_second_seed']==[]
        if key=='color_sensitivity':
            for arm,body in data['arms'].items():
                assert body['identity_predictions_exact'] and len(body['rows'])==5
                for path,digest in body['bindings'].items():
                    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
                cost['cell_'+arm]=dict(timing=body['timing'],scope='canonical center crop, warm PNG+preprocess+encoder; CTransPath is not native XCell')
    requirements=[
        dict(id='P1I_matched_three_frozen_encoders_and_strata',state='evidence_present',evidence=['image_resnet','image_phikon','image_sam','image_comparison']),
        dict(id='P1I_one_limited_joint_recipe',state='complete_negative',evidence=['joint_comparison'],decision='retain frozen candidate, no secondseed or LR/layer sweep'),
        dict(id='P1I_extraction_cache_cost',state='historical_cost_available',evidence=['image_resnet','image_phikon','image_sam'],limitation='no isolated online interaction or fullWSI cost claim'),
        dict(id='P1C_common_crop_probe_shuffle_retrieval',state='evidence_present_with_scope',evidence=['cell_probe'],limitation='two BC/TC expert labels only; point CTransPath not native XCell'),
        dict(id='P1C_source_dependence',state='fixed_prediction_evidence_present',evidence=['cell_uncertainty'],limitation='not independent seed or external confirmation'),
        dict(id='P1C_same_tissue_instances_downstream',state='complete_negative',evidence=['cell_two_seed','cell_frozen_revision'],decision='stop residual recipes; no tissue replacement claim'),
        dict(id='P1C_color_and_center_crop_cost',state='synthetic_evidence_present',evidence=['color_sensitivity'],limitation='no real external stain domain'),
        dict(id='P1C_native_XCell_cost_comparable_ledger',state='missing',evidence=['native_XCell_gate'],limitation='input correctness accepted; canonical CTransPath timing cannot substitute mask/context native XCell cost'),
        dict(id='P1_candidate_freeze_and_transfer_selection',state='pending_synthesis',limitation='Phikon working image candidate; LEMON probe advantage does not justify unqualified tissue replacement'),
    ]
    report=dict(status='complete_P1_obligation_audit_with_gaps',evidence=evidence,cost_ledger=cost,requirements=requirements,
                P1_complete=False,fullWSI_promoted=False,next_action='measure native XCell context pipeline cost separately, then freeze candidate/input choice with explicit negative scope',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],P1_complete=False,gaps=[r['id'] for r in requirements if r['state'] in ['missing','pending_synthesis']])))


if __name__=='__main__':main()
