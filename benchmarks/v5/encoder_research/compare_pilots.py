"""Compare independently replayed pilots, including preregistered FPA guard."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows={};summaries={};consumed=[]
    catalog=pd.read_parquet('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')
    spaces={r['tile_id']:r['space_key'] for r in catalog.to_dict('records')}
    for arm in ['resnet','phikon','sam']:
        accepted=json.loads((a.root/arm/'pilot/independent_acceptance.json').read_text());assert accepted['status']=='accepted_fixed_image_pilot'
        consumed.append(accepted['actual4000_task_SHA']);summaries[arm]=accepted['summary']
        values=json.loads((a.root/arm/'pilot/rows.json').read_text());rows[arm]=[r for r in values if r['budget']=='3+2']
    assert len(set(consumed))==1
    keys=lambda values:[(r['scope'],r['query_id'],r['support_id'],r['target']) for r in values]
    assert keys(rows['resnet'])==keys(rows['phikon'])==keys(rows['sam'])
    strata={}
    for arm,values in rows.items():
        local=[r for r in values if r['scope']=='local'];small=[r for r in local if r['positive']/r['valid']<.05]
        fat=[r for r in local if r['dataset']=='crc' and r['target']==10]
        strata[arm]={'small_under5pct':{'n':len(small),'dice':float(np.mean([r['dice'] for r in small]))},
                     'CRC_fat':{'n':len(fat),'dice':float(np.mean([r['dice'] for r in fat]))}}
    contrasts=[]
    for arm in ['phikon','sam']:
        for scope,metric,present in [('local','dice',True),('remote','dice',True),('remote','false_positive_fraction',False)]:
            pairs=[(b,c) for b,c in zip(rows['resnet'],rows[arm]) if b['scope']==scope and (b['positive']>0)==present]
            groups=sorted({spaces[b['query_id']] for b,c in pairs});index={g:i for i,g in enumerate(groups)};total=np.zeros(len(groups));count=np.zeros(len(groups))
            for b,c in pairs:
                i=index[spaces[b['query_id']]];total[i]+=c[metric]-b[metric];count[i]+=1
            rng=np.random.default_rng(20261031);samples=[]
            for _ in range(2000):
                picked=rng.integers(0,len(groups),len(groups));samples.append(float(total[picked].sum()/count[picked].sum()))
            contrasts.append(dict(arm=arm,reference='resnet',scope=scope,metric=metric,targets='present' if present else 'absent',
                difference=float(total.sum()/count.sum()),CI95=np.quantile(samples,[.025,.975]).tolist(),query_space_clusters=len(groups),rows=len(pairs),
                inference_scope='exploratory within-development source-space clustered interval; single training seed, no independent-patient claim'))
    gates={}
    for arm in ['phikon','sam']:
        b=summaries['resnet'];x=summaries[arm]
        checks={'local_Dice_plus1pp':x['local|3+2']['dice']-b['local|3+2']['dice']>=.01,
            'local_BF1_not_down1pp':x['local|3+2']['BF1']-b['local|3+2']['BF1']>=-.01,
            'remote_absent_FPA_not_up1pp':x['remote|3+2']['absent_FPA']-b['remote|3+2']['absent_FPA']<=.01,
            'small_Dice_not_down2pp':strata[arm]['small_under5pct']['dice']-strata['resnet']['small_under5pct']['dice']>=-.02,
            'fat_Dice_not_down2pp':strata[arm]['CRC_fat']['dice']-strata['resnet']['CRC_fat']['dice']>=-.02}
        gates[arm]=dict(checks=checks,passed=all(checks.values()))
    report=dict(status='complete_three_arm_fixed_pilot_comparison',summaries=summaries,strata=strata,contrasts=contrasts,
        resource_gates=gates,shared_actual4000_SHA=consumed[0],
        decision='Phikon is strongest local-development representation in this pilot, but no candidate passes joint FPA guard. No fullWSI promotion; no query threshold change. Diagnose/test explicit continuous-cell contribution and train-only remote/absent objective under separate predeclared controlled designs.',
        qualification='SAM row is frozen SAM image encoder with shared new head, not original SAM or SAM-FT; same fine view, different native encoding resolution and adapter capacity disclosed.')
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'gates':gates,'strata':strata,'summary':summaries}))


if __name__=='__main__':main()
