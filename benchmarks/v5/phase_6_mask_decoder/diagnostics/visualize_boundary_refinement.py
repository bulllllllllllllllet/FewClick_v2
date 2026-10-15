"""Render paired boundary development results and cluster bootstrap deltas."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); root=a.run
    summary=json.loads((root/'summary.json').read_text())
    run=json.loads((root/'run.json').read_text())
    inputs=np.load(root/'validation_inputs.npz',allow_pickle=False)
    baseline=json.loads((root/'baseline_rows.json').read_text())
    groups={}
    for i,row in enumerate(baseline): groups.setdefault(row['patch_id'],[]).append(i)
    cluster=list(groups.values()); rng=np.random.default_rng(20261014)
    paired={}; arms={}; rows={}
    for arm in ('pixel','boundary'):
        epoch=summary['arms'][arm]['best_boundary']['epoch']
        rows[arm]=json.loads((root/f'{arm}_epoch_{epoch:03}_rows.json').read_text())
        arms[arm]=np.load(root/f'{arm}_best_predictions.npz',allow_pickle=False)['probability']
        paired[arm]={'epoch':epoch,'eligible':summary['arms'][arm]['best_boundary']['summary']['eligible'],
                     'selection':'best validation BF1@2; same validation used for selection and diagnosis', 'deltas':{},'by_target_class':{}}
        for key in ('dice','bf1_2','boundary_band_iou_2'):
            diff=np.array([r[key]-b[key] if r[key] is not None and b[key] is not None else np.nan for r,b in zip(rows[arm],baseline,strict=True)])
            draws=[]
            for _ in range(2000):
                ids=np.concatenate([cluster[k] for k in rng.integers(0,len(cluster),len(cluster))])
                if np.isfinite(diff[ids]).any(): draws.append(float(np.nanmean(diff[ids])))
            paired[arm]['deltas'][key]={'mean':float(np.nanmean(diff)) if np.isfinite(diff).any() else None,
                'patch_bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist() if draws else None}
        for cls in sorted({r['target_class'] for r in baseline}):
            ids=[i for i,r in enumerate(baseline) if r['target_class']==cls]
            item={'episodes':len(ids)}
            for key in ('dice','bf1_2','boundary_band_iou_2'):
                diffs=[rows[arm][i][key]-baseline[i][key] for i in ids
                       if rows[arm][i][key] is not None and baseline[i][key] is not None]
                item[key+'_delta']=float(np.mean(diffs)) if diffs else None
                item[key+'_evaluable']=len(diffs)
            paired[arm]['by_target_class'][str(cls)]=item
    paired['scope']='exploratory spatial validation; selection on same val; historical encoder exposure; CIs not independent confirmatory evidence'
    (root/'paired_report.json').write_text(json.dumps(paired,indent=2))
    # Show best gains, largest regressions, and typical deterministic samples.
    delta=np.array([r['bf1_2']-b['bf1_2'] if r['bf1_2'] is not None and b['bf1_2'] is not None else 0 for r,b in zip(rows['boundary'],baseline,strict=True)])
    order=np.argsort(delta)
    ids=list(dict.fromkeys([int(order[0]),int(order[-1]),int(order[len(order)//2]),0]))[:4]
    fig,axes=plt.subplots(len(ids),5,figsize=(17,3.7*len(ids)),squeeze=False)
    for row,i in enumerate(ids):
        image=np.clip(inputs['image'][i].transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
        truth=inputs['truth'][i]&inputs['valid'][i]
        ax=axes[row,0]; ax.imshow(image); ax.set_title(f"H&E: class {baseline[i]['target_class']}\n{baseline[i]['patch_id']}",fontsize=8)
        for prefix,color in (('positive','lime'),('negative','red')):
            # Prompt metadata is saved separately; no GT-driven replacement clicks.
            prompts=run.get('validation_prompts',[])
            if prompts:
                xy=np.array(prompts[i][prefix])
                if len(xy): ax.scatter(xy[:,0]*image.shape[1],xy[:,1]*image.shape[0],s=28,c=color,marker='+')
        axes[row,1].imshow(truth,cmap='gray',vmin=0,vmax=1); axes[row,1].set_title('Target GT (ignore excluded)')
        for col,(arm,prob) in enumerate((('baseline',inputs['baseline'][i]),('pixel',arms['pixel'][i]),('boundary',arms['boundary'][i])),start=2):
            ax=axes[row,col]; ax.imshow(image)
            valid=inputs['valid'][i]; pred=(prob>=.5)&valid
            if truth.any() and (~truth).any(): ax.contour(truth,levels=[.5],colors=['lime'],linewidths=.8)
            if pred.any() and (~pred).any(): ax.contour(pred,levels=[.5],colors=['red'],linewidths=.8)
            data=baseline[i] if arm=='baseline' else rows[arm][i]
            bf=data['bf1_2']; ax.set_title(f"{arm}: Dice={data['dice']:.3f}, BF1={bf if bf is None else round(bf,3)}")
        for ax in axes[row]: ax.axis('off')
    fig.suptitle('Green: target GT contour; red: prediction contour. Rows: regression / gain / median / first.',fontsize=11)
    fig.tight_layout(); fig.savefig(root/'boundary_contact_sheet.png',dpi=150); plt.close(fig)
    print(root/'boundary_contact_sheet.png')


if __name__=='__main__': main()
