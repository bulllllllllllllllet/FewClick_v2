#!/usr/bin/env python3
"""Audit paired manifests and summarize all completed Panel1 arms, including losses."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def values(row):
    m=row['metrics']
    result={'pixel_macro_dice':m['segmentation']['macro_dice'],'pixel_macro_iou':m['segmentation']['macro_iou'],
            'crc_val_ce':m['crc_val_ce']}
    result.update({f'click{b}_dice':m['clicks'][str(b)]['task_macro_dice'] for b in (2,4,6)})
    for t in ('0.8','0.9'):
        result[f'noc{t}']=m['noc'][t]['task_macro_noc']
        result[f'fail{t}']=m['noc'][t]['task_macro_failure_rate']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--timestamp',required=True)
    args=p.parse_args()
    rows=[];manifests=[];audits=[]
    for root in args.roots:
        status=json.loads((root/'status.json').read_text())
        if status['status']!='complete':raise ValueError(f'incomplete: {root}')
        manifest=json.loads((root/'manifest.json').read_text());manifests.append(manifest)
        results=json.loads((root/'results.json').read_text())
        if len({r['initial_cell_hash'] for r in results})!=1:raise AssertionError('initializations differ')
        for seed in {r['seed'] for r in results}:
            group=[r for r in results if r['seed']==seed]
            if len({r['main_schedule_hash'] for r in group})!=1:raise AssertionError('main training schedule mismatch')
            baseline=next(r for r in group if r['arm']=='seg_only')
            initial_prompts={}
            for path in sorted((root/f'{seed}_seg_only/evaluation').glob('click_[0-9]*.json')):
                initial_prompts[path.name]=json.loads(path.read_text())['clicks_y_x_sign']['2']
            for r in group:
                directory=root/f"{seed}_{r['arm']}/evaluation"
                observed={path.name:json.loads(path.read_text())['clicks_y_x_sign']['2'] for path in directory.glob('click_[0-9]*.json')}
                if observed!=initial_prompts:raise AssertionError('initial click/task sets differ across arms')
                row=dict(seed=seed,arm=r['arm'],**values(r))
                row['delta_vs_seg_only']={k:v-values(baseline)[k] for k,v in values(r).items()}
                rows.append(row)
            audits.append(dict(root=str(root),seed=seed,shared_initial_click_tasks=len(initial_prompts),
                               checkpoint_count=len(group),initialization_and_schedule_match=True))
    if len({m['fixed_manifest_hash'] for m in manifests})!=1:raise AssertionError('data differs across seeds')
    if len({m['runner_sha256'] for m in manifests})!=1 or len({m['helper_sha256'] for m in manifests})!=1:
        raise AssertionError('source code differs across seeds')
    arms=[a for a in ('seg_only','joint_003','joint_010','panel1_pretrain','seg_budget') if a in {r['arm'] for r in rows}]
    summary={}
    for arm in arms:
        group=[r for r in rows if r['arm']==arm]
        summary[arm]={}
        for key in values(json.loads((args.roots[0]/'results.json').read_text())[0]):
            x=np.array([r[key] for r in group]); delta=np.array([r['delta_vs_seg_only'][key] for r in group])
            summary[arm][key]={'mean':float(x.mean()),'seed_sd':float(x.std(ddof=1)) if len(x)>1 else None,
                               'mean_delta_vs_seg_only':float(delta.mean()),'per_seed':x.tolist()}
    out=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')/f'panel1_controlled_report_{args.timestamp}'
    out.mkdir(parents=True,exist_ok=False)
    data={'scope':'historical validation; no test; fine-only frozen Phase5/6; oracle actual pixel clicks; budget control matches updates not FLOPs',
          'summary':summary,'paired_rows':rows,'audits':audits,'fixed_manifest_hash':manifests[0]['fixed_manifest_hash'],
          'crc_sampling':{s:{'cases':dict(Counter(r['case_id'] for r in manifests[0]['crc']['selected'][s])),
                             'spatial_groups':len({r['spatial_group_id'] for r in manifests[0]['crc']['selected'][s]})} for s in ('train','val')}}
    (out/'summary.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    lines=['# Panel1 controlled transfer','',data['scope'],'',
           '| arm | pixel Dice | pixel IoU | 2 clicks Dice | 4 clicks Dice | 6 clicks Dice | NoC@0.8 | failure@0.8 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        s=summary[arm]
        lines.append('| '+arm+' | '+' | '.join(f"{s[k]['mean']:.6f}" for k in ('pixel_macro_dice','pixel_macro_iou','click2_dice','click4_dice','click6_dice','noc0.8','fail0.8'))+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    # Prespecified illustration: first eligible class of validation patch zero, seed17.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=args.roots[0];seed=json.loads((root/'results.json').read_text())[0]['seed']
    first=sorted((root/f'{seed}_seg_only/evaluation').glob('click_000_*.npz'))
    if first:
        name=first[0].name;gt=np.load(first[0])['gt']
        fig,axes=plt.subplots(len(arms),5,figsize=(15,3*len(arms)))
        rgb=np.asarray(Image.open(root/'val_000_he.png'))
        for i,arm in enumerate(arms):
            path=root/f'{seed}_{arm}/evaluation'/name
            saved=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
            axes[i,0].imshow(rgb);axes[i,0].contour(gt,levels=[.5],colors=['yellow'],linewidths=.5)
            axes[i,0].set_title(f'{arm}\nHE + target boundary')
            axes[i,1].imshow(gt,vmin=0,vmax=1,cmap='gray');axes[i,1].set_title('Target ground truth')
            for j,b in enumerate((2,4,6),2):
                axes[i,j].imshow(saved[f'pred_{b}'],vmin=0,vmax=1,cmap='gray')
                for y,x,sign in meta['clicks_y_x_sign'][str(b)]:
                    axes[i,j].scatter(x,y,c='lime' if sign else 'red',s=18,marker='+' if sign else 'x')
                dice=next(r['dice'] for r in meta['curve'] if r['budget']==b)
                axes[i,j].set_title(f'{b} clicks; Dice={dice:.3f}')
            for ax in axes[i]:ax.axis('off')
        fig.suptitle(f'Fixed example: seed {seed}, val patch 0, class {meta["target_class"]}; green + / red - clicks')
        fig.tight_layout();fig.savefig(out/'fixed_example.png',dpi=140);plt.close(fig)
    print(json.dumps({'output':str(out),'summary':summary},indent=2))


if __name__=='__main__':main()
