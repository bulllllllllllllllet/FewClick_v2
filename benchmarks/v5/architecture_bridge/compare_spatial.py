"""Bind six complete P3 runs before reporting preregistered contrasts."""
import argparse
import json
from pathlib import Path
import torch
from .spatial_controls import ARMS
from .spatial_quality_guards import compare_rows
from .compare_vocabulary import summarize_arm
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS,sha
from benchmarks.v5.encoder_research.compare_continuous_pilots import paired_interval

PAIRS=[('S11','S00'),('S10','S00'),('S01','S00'),('S11','S10'),('S11','S01'),('S00','I'),('S11','I'),('H11','S11')]


def load_runs(root):
    assert json.loads((root/'stage.json').read_text())['status']=='complete','Six formal runs must finish first'
    contract=json.loads((root/'contract.json').read_text());assert contract.get('updates')==2000 and contract.get('batch')==4,'A formal 2000x4 contract is required'
    assert contract['seed'] in [20261103,20261104]
    trainer='benchmarks/v5/architecture_bridge/train_spatial.py'
    reports={};rows={};evidence={}
    for arm in ARMS:
        folder=root/arm;r=json.loads((folder/'report.json').read_text());proof=json.loads((folder/'independent_acceptance.json').read_text())
        assert proof['status']=='accepted_P3_spatial_evaluation' and not proof['smoke'] and not r['smoke']
        assert proof['arm']==r['arm']==arm and proof['seed']==r['seed']==contract['seed']
        assert proof['rows']==3711 and r['updates']==2000 and r['consumed_tasks']==8000
        assert r['source_sha256']==contract['sources'][trainer] and r['fresh_prediction_replay_count']==12
        for name,key in [('report.json','report_sha256'),('checkpoint.pt','checkpoint_sha256'),('rows.json','rows_sha256'),('masks.npz','masks_sha256')]:
            assert sha(folder/name)==proof[key],(arm,name)
        assert r['checkpoint_sha256']==proof['checkpoint_sha256']
        assert sha(folder/'training_consumed.jsonl')==contract['consumption_sha256']
        assert r['protocol_sha256']==contract['protocol_sha256']
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['updates']==2000 and checkpoint['arm']==arm and checkpoint['seed']==contract['seed']
        assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
        states=list(checkpoint['optimizer']['state'].values());assert states and all(int(s['step'])==2000 for s in states)
        assert all(g['lr']==g['weight_decay']==1e-4 for g in checkpoint['optimizer']['param_groups'])
        del checkpoint
        reports[arm]=r;rows[arm]=json.loads((folder/'rows.json').read_text())
        assert len(rows[arm])==3711
        evidence[arm]={n:sha(folder/n) for n in ['independent_acceptance.json','report.json','rows.json','masks.npz','training_consumed.jsonl']}
    for key in ['seed','inputs','initial_state_sha256','protocol_sha256','source_sha256']:
        assert all(reports[x][key]==reports['S11'][key] for x in ARMS),key
    return rows,evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows,evidence=load_runs(a.run);_,inventory=load_cell_catalog(TASKS)
    values={};flat={}
    for arm in ARMS:flat[arm],_,values[arm]=summarize_arm(rows[arm])
    contrasts={}
    for candidate,reference in PAIRS:
        value=compare_rows(rows[reference],rows[candidate]);value['remote_macro_query_space_interval']=paired_interval(flat[reference],flat[candidate],inventory,'remote','dice',True,True)
        contrasts[candidate+'-'+reference]=value
    eligible=next((pair for pair in ['S11-S00','S01-S00','S10-S00'] if contrasts[pair]['pass_all']),None)
    interaction={key:(values['S11'][key]-values['S01'][key])-(values['S10'][key]-values['S00'][key]) for key in values['S00']}
    result=dict(status='accepted_P3_complete_six_arm_comparison',values=values,contrasts=contrasts,descriptive_interaction=interaction,
                prioritized_replication_pair=eligible,evidence=evidence,contract_sha256=sha(a.run/'contract.json'),source_sha256=sha(__file__),
                P3_complete=False,WSI_promoted=False,limitations=['exposed development regions; conditional query-space intervals retain fixed support and patient dependence',
                'one formal seed per table; replication selection does not prove effect',
                'H11 leaves area metadata and cell-set context; image-only has inactive cell parameters',
                'mass intervention jointly changes attention prior and region density'])
    a.output.mkdir(exist_ok=False);(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
