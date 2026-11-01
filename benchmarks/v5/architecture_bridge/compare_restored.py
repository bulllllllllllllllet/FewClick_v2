"""All fixed bridge contrasts after complete restored-objective acceptance."""
import argparse,json,os
from collections import defaultdict
from pathlib import Path
from benchmarks.v5.encoder_research.accept_continuous_pilot import summarize
from benchmarks.v5.encoder_research.compare_continuous_pilots import summary_values,resource_guards,paired_interval
from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
from benchmarks.v5.encoder_research.continuous_data import TASKS
from benchmarks.v5.encoder_research.build_joint_task_manifest import sha

BOUNDS=dict(primary_Dice_gain=.01,remote_BF1_max_drop=.01,absent_FPA_max_increase=.01,local_Dice_max_drop=.02,small_Dice_max_drop=.02,fat_Dice_max_drop=.02)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert json.loads((a.run/'stage.json').read_text())['status']=='complete'
    rows={};values={};summaries={};evidence={};reports={}
    for arm in ['A','B','C','D']:
        folder=a.run/arm;report=json.loads((folder/'report.json').read_text());accept=json.loads((folder/'independent_acceptance.json').read_text())
        assert accept['status']=='accepted_restored_bridge_evaluation' and not accept['smoke'] and accept['rows']==3711
        assert accept['report_sha256']==sha(folder/'report.json')
        assert accept['checkpoint_sha256']==report['checkpoint_sha256']==sha(folder/'checkpoint.pt')
        assert report['updates']==2000 and report['consumed_tasks']==8000 and report['epochs']==2
        assert (folder/'training_consumed.jsonl').read_text()==(a.run/'A/training_consumed.jsonl').read_text()
        flat=[dict(r['plan'],scope=r['scope'],budget=r['budget'],**r['metrics']) for r in json.loads((folder/'rows.json').read_text())];assert len(flat)==3711
        groups=defaultdict(list)
        for r in flat:groups[r['scope']+'|'+r['budget']].append(r)
        local=groups['local|3+2'];summary=dict(summary={k:summarize(v) for k,v in groups.items()},local_strata=dict(small=summarize([r for r in local if r['positive']/r['valid']<.05]),fat=summarize([r for r in local if r['dataset']=='crc' and r['target']==10])))
        rows[arm]=flat;summaries[arm]=summary;values[arm]=summary_values(summary);reports[arm]=report
        evidence[arm]={n:sha(folder/n) for n in ['report.json','independent_acceptance.json','rows.json','masks.npz','training_consumed.jsonl']}
    for arm in ['B','C','D']:
        for key in ['training_sha256','inputs','source_sha256','gate_sha256']:assert reports[arm][key]==reports['A'][key]
        for l,r in zip(rows['A'],rows[arm]):
            for key in ['scope','budget','query_id','support_id','target','kind','positive_xy','negative_xy','positive','valid']:assert l[key]==r[key]
    _,inventory=load_cell_catalog(TASKS);contrasts={}
    for candidate,reference in [('B','A'),('C','A'),('D','A'),('D','B'),('D','C')]:
        contrasts[candidate+'-'+reference]=dict(guards=resource_guards(values[reference],values[candidate],BOUNDS),
            primary_interval=paired_interval(rows[reference],rows[candidate],inventory,'remote','dice',True,macro=True))
    result=dict(status='accepted_restored_bridge_four_arm_comparison',worker_pid=os.getpid(),values=values,summaries=summaries,contrasts=contrasts,bounds=BOUNDS,evidence=evidence,
        interaction_D_minus_B_minus_C_plus_A={k:values['D'][k]-values['B'][k]-values['C'][k]+values['A'][k] for k in values['A']},
        seed2_eligible=[arm for arm in ['B','C','D'] if contrasts[arm+'-A']['guards']['pass_all']],
        fullWSI_promoted=False,source_sha256=sha(__file__),limitations=['one seed on exposed development regions','same support/query reuse; conditional query-space intervals do not prove independent generalization','fixed input views and update counts, unequal frozen backbone compute/pretraining','TIGER3 absent-only; per-target NA retained','retainedJ5 history not an equal-supervision main arm'])
    a.output.mkdir(exist_ok=False);(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(values=values,guards={k:v['guards']['tests'] for k,v in contrasts.items()},seed2_eligible=result['seed2_eligible']),indent=2))


if __name__=='__main__':main()
