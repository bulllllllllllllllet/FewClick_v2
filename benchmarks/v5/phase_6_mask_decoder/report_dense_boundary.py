"""Report completed paired pilot; validation selection remains developmental."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .probe_boundary_refinement import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    if json.loads((a.run/'status.json').read_text())['status']!='complete':raise RuntimeError('training incomplete')
    destination=a.run/'paired_report_1014_190217.json'
    if destination.exists():raise FileExistsError(destination)
    summary=json.loads((a.run/'summary.json').read_text());run=json.loads((a.run/'run.json').read_text())
    base=json.loads((a.run/'baseline_rows.json').read_text())
    manifest=pd.read_parquet(Path(run['args']['spatial_data'])/'spatial_manifest.parquet').set_index('patch_id')
    ids=[r['patch_id'] for r in base];groups=[str(manifest.loc[k,'spatial_group_id']) for k in ids]
    group_ids=sorted(set(groups));member=[np.flatnonzero(np.asarray(groups)==g) for g in group_ids]
    rng=np.random.default_rng(20261014);draws=rng.integers(0,len(member),(2000,len(member)))
    report={'scope':'developmental spatial-block validation; same val used for epoch selection; historical encoder exposure',
            'val_episodes':len(base),'spatial_blocks':len(member),'baseline':summary['baseline'],'arms':{},'test_used':False}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,arm in summary['arms'].items():
        best=max(arm['history'],key=lambda h:h['bf1_2']);epoch=best['epoch']
        rows=json.loads((a.run/f'{name}_epoch_{epoch:03}_rows.json').read_text())
        if [r['patch_id'] for r in rows]!=ids:raise RuntimeError('pairing mismatch')
        result={'diagnostic_best_bf1_epoch':epoch,'summary':best,'eligible_epochs':[h['epoch'] for h in arm['history'] if h['eligible']],
                'per_class':{},'paired_intervals':{}}
        for metric in ['dice','bf1_2','boundary_band_iou_2']:
            difference=np.asarray([r[metric]-b[metric] for r,b in zip(rows,base)])
            sums=np.asarray([difference[m].sum() for m in member]);counts=np.asarray([len(m) for m in member])
            samples=sums[draws].sum(1)/counts[draws].sum(1)
            result['paired_intervals'][metric]={'mean_delta':float(difference.mean()),'spatial_block_bootstrap_95':np.percentile(samples,[2.5,97.5]).tolist()}
        for c in sorted(set(r['target_class'] for r in rows)):
            rr=[r for r in rows if r['target_class']==c];bb=[r for r in base if r['target_class']==c]
            result['per_class'][str(c)]={'episodes':len(rr),'dice_delta':float(np.mean([r['dice']-b['dice'] for r,b in zip(rr,bb)])),
                                       'bf1_2_delta':float(np.mean([r['bf1_2']-b['bf1_2'] for r,b in zip(rr,bb)]))}
        regression_ids={r['identity']['patch_id'] for r in run['refs']['val'] if r.get('regression')}
        result['regression_subset']=summarize([r for r in rows if r['patch_id'] in regression_ids])
        report['arms'][name]=result
        for ax,key in zip(axes,['bf1_2','dice']):
            ax.plot([h['epoch']+1 for h in arm['history']],[h[key] for h in arm['history']],'-o',label=name)
    for ax,key in zip(axes,['bf1_2','dice']):
        ax.axhline(summary['baseline'][key],ls='--',color='black',label='frozen J5');ax.set(xlabel='Epoch',ylabel=key);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Paired validation curves; developmental checkpoint selection');fig.tight_layout();fig.savefig(a.run/'learning_curves_1014_190217.png',dpi=160);plt.close(fig)
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
