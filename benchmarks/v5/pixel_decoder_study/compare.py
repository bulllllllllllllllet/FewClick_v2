"""Bind both final runs before evaluating fixed localization and safety criteria."""
import argparse
import json
from pathlib import Path

import torch

from .model import ARMS
from benchmarks.v5.architecture_bridge.spatial_quality_guards import compare_rows
from benchmarks.v5.architecture_bridge.compare_vocabulary import summarize_arm
from benchmarks.v5.encoder_research.continuous_prompt import tree_sha
from benchmarks.v5.encoder_research.continuous_data import sha, TASKS
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.compare_continuous_pilots import paired_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    torch.set_num_threads(2)
    assert json.loads((root/'stage.json').read_text())['status']=='complete_runs'
    c=json.loads((root/'contract.json').read_text()); rows={};reports={};evidence={}
    assert c['updates']==2000 and c['batch']==4 and c['seed']==20261103
    for arm in ARMS:
        folder=root/arm;proof=json.loads((folder/'independent_acceptance.json').read_text());r=json.loads((folder/'report.json').read_text())
        assert proof['status']=='accepted_P3_spatial_evaluation' and not proof['smoke'] and proof['arm']==arm
        assert r['updates']==2000 and r['consumed_tasks']==8000 and r['fresh_prediction_replay_count']==12
        assert r['execution_contract_sha256']==sha(root/'contract.json')
        assert r['initial_common_sha256']==c['initial_common_sha256']
        for filename,key in [('report.json','report_sha256'),('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256')]:
            assert sha(folder/filename)==proof[key]
        assert sha(folder/'training_consumed.jsonl')==c['consumption_sha256']
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['updates']==2000 and checkpoint['arm']==arm and checkpoint['seed']==c['seed']
        assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
        assert all(int(s['step'])==2000 for s in checkpoint['optimizer']['state'].values())
        assert all(g['lr']==g['weight_decay']==1e-4 for g in checkpoint['optimizer']['param_groups'])
        if arm=='region':
            original=torch.load(Path(c['old_formal'])/'I/checkpoint.pt',map_location='cpu',weights_only=True)
            common={k.removeprefix('base.'):v for k,v in checkpoint['model'].items()}
            assert tree_sha(common)==tree_sha(original['model']), 'Matched control must reproduce original final I state'
            assert tree_sha(checkpoint['optimizer'])==tree_sha(original['optimizer'])
            assert sha(folder/'masks.npz')==sha(Path(c['old_formal'])/'I/masks.npz')
            del original,common
        del checkpoint
        reports[arm]=r;rows[arm]=json.loads((folder/'rows.json').read_text());assert len(rows[arm])==3711
        evidence[arm]={name:sha(folder/name) for name in ['independent_acceptance.json','report.json','masks.npz','training_consumed.jsonl']}
    for field in ['seed','inputs','initial_common_sha256','source_sha256','protocol_sha256']:
        assert reports['region'][field]==reports['pixel'][field]
    flat={};values={}
    for arm in ARMS:flat[arm],_,values[arm]=summarize_arm(rows[arm])
    guards=compare_rows(rows['region'],rows['pixel'])
    changes=guards['original_six']['candidate_minus_reference']
    localization=dict(local_small_Dice_gain_5pp=changes['small_Dice']>=.05,
                      local_BF1_gain_1pp=changes['local_BF1']>=.01)
    _,inventory=load_cell_catalog(TASKS)
    intervals={scope:paired_interval(flat['region'],flat['pixel'],inventory,scope,'dice',True,True) for scope in ['local','remote']}
    result=dict(status='accepted_matched_pixel_decoder_comparison',values=values,safety_guards=guards,
                localization_tests=localization,localization_pass=all(localization.values()),
                first_seed_joint_candidate=guards['pass_all'] and all(localization.values()),
                conditional_query_space_intervals=intervals,evidence=evidence,legacy_control_final_state_exact=True,
                contract_sha256=sha(root/'contract.json'),source_sha256=sha(__file__),P3_complete=False,WSI_promoted=False,
                limitation='one seed, exposed development set; added decoder parameters and same-image RGB path vary together; no auto seed2')
    out=root/'comparison';out.mkdir(exist_ok=False);(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
