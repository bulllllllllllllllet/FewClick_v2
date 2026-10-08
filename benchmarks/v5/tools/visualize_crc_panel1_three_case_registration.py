#!/usr/bin/env python3
"""One-sheet H&E/DAPI registration review for three CRC Panel1 cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import openslide
import pandas as pd

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES


ARTIFACT = Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
CASE_MAP = {'2017-23800D':'17-23800D', '2019-00538F':'17-00538F', '2022-39941E':'22-39941E'}
PICKS = [
    ('2017-23800D','strong','x20480_y55808'),
    ('2017-23800D','lower','x23040_y18944'),
    ('2019-00538F','strong','x21504_y16896'),
    ('2019-00538F','lower','x13824_y15872'),
    ('2022-39941E','strong','x36864_y64000'),
    ('2022-39941E','lower','x25088_y20992'),
]
COLORS = ['#facc15','#f97316','#38bdf8','#a78bfa','#4ade80']


def stretch(array: np.ndarray, low=.5, high=99.7) -> np.ndarray:
    lo, hi = np.percentile(array, [low, high])
    return np.clip((array.astype(float)-lo)/max(hi-lo, 1.),0,1)


def choose(labels: np.ndarray, limit=14) -> np.ndarray:
    rng=np.random.default_rng(20261008)
    picks=[]
    for klass in range(len(CLASSES)):
        ids=np.flatnonzero(labels==klass)
        picks.extend(rng.choice(ids,size=min(2,len(ids)),replace=False).tolist())
    remainder=np.setdiff1d(np.arange(len(labels)),np.asarray(picks,dtype=int))
    if len(picks)<limit and len(remainder):
        picks.extend(rng.choice(remainder,size=min(limit-len(picks),len(remainder)),replace=False).tolist())
    return np.array(sorted(picks),dtype=int)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-rows',type=int,default=6)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ARTIFACT):
        p.error('output outside FewClick artifact root')
    if not 1 <= a.max_rows <= 6:
        p.error('max rows must be 1..6')
    a.output.mkdir(parents=True,exist_ok=False)
    fig, axes=plt.subplots(a.max_rows,3,figsize=(12,3.45*a.max_rows),squeeze=False)
    detail=[]
    slide_cache={}
    try:
        for i,(case,tier,patch) in enumerate(PICKS[:a.max_rows]):
            root=ARTIFACT/f'panel1_three_case_full_0924_134324_{case}'
            audit=json.loads((root/'audit.json').read_text())
            risk=pd.read_csv(ARTIFACT/'panel1_candidate_integrity_cache_full_1008_135230'/f'{case}_patch_risk.csv')
            row=risk.loc[risk.patch.eq(patch)]
            if len(row)!=1:
                raise ValueError(f'patch risk row missing/duplicate: {case} {patch}')
            row=row.iloc[0]
            short=CASE_MAP[case]
            if case not in slide_cache:
                he=openslide.OpenSlide(audit['registered_he_wsi'])
                mif=list((Path('/nfs-medical6/data/SRRSH/crc/panel_1/mIF')/f'{short}(FAP)'/'Scan1').glob('*.qptiff'))
                if len(mif)!=1:
                    raise ValueError(f'expected one mIF QPTIFF for {case}')
                slide_cache[case]=(he,openslide.OpenSlide(str(mif[0])))
            he_slide,mif_slide=slide_cache[case]
            x,y=map(int,patch.replace('x','').replace('y','').split('_'))
            he=np.asarray(he_slide.read_region((x,y),0,(512,512)).convert('RGB'))
            dapi_raw=np.asarray(mif_slide.read_region((x,y),0,(512,512)).convert('RGB'))[:,:,0]
            if he.shape!=(512,512,3) or dapi_raw.shape!=(512,512):
                raise ValueError('source region is not 512px')
            with np.load(root/'patches'/f'{patch}.npz',allow_pickle=False) as z:
                he_xy=z['he_xy'].copy();mif_xy=z['mif_global_xy'].copy()-np.array([x,y]);
                labels=z['labels'].copy();dist=z['match_distance_px'].copy();
                he_ids=z['he_ids'].copy();mif_ids=z['mif_ids'].copy()
            if len(labels)!=int(row.candidate_cells) or np.any((he_xy<0)|(he_xy>=512)) or np.any((mif_xy<0)|(mif_xy>=512)):
                raise ValueError('cache/patch coordinates or count mismatch')
            if not np.allclose(np.linalg.norm(he_xy-mif_xy,axis=1),dist,atol=.01):
                raise ValueError('cached match distance mismatch')
            selected=choose(labels)
            dapi=stretch(dapi_raw)
            he_gray=(.299*he[:,:,0]+.587*he[:,:,1]+.114*he[:,:,2])/255.
            he_nuc=stretch(1-he_gray)
            overlay=np.stack((he_nuc, dapi, he_nuc),axis=-1)
            panels=[he,np.stack((dapi,)*3,axis=-1),overlay]
            for j,ax in enumerate(axes[i]):
                ax.imshow(panels[j],interpolation='nearest')
                ax.set_xlim(0,511);ax.set_ylim(511,0);ax.set_xticks([]);ax.set_yticks([])
                if j<2:
                    ax.scatter(mif_xy[selected,0],mif_xy[selected,1],s=38,
                               c=[COLORS[k] for k in labels[selected]],edgecolors='black',linewidths=.5,zorder=4)
                    ax.scatter(he_xy[selected,0],he_xy[selected,1],s=82,facecolors='none',
                               edgecolors='white',linewidths=1.1,zorder=5)
                title=(f'{case} · {tier} · {patch} · {len(labels)} cells · ratio {row.real_to_shift_ratio:.2f}\nRegistered H&E'
                       if j==0 else ('mIF DAPI (contrast stretched)' if j==1 else 'Overlay: magenta H&E / green DAPI'))
                ax.set_title(title,fontsize=9)
            for n in selected:
                detail.append({'case_id':case,'tier':tier,'patch':patch,'he_id':int(he_ids[n]),
                               'mif_id':int(mif_ids[n]),'label':CLASSES[int(labels[n])],
                               'he_x':float(he_xy[n,0]),'he_y':float(he_xy[n,1]),
                               'mif_x_local':float(mif_xy[n,0]),'mif_y_local':float(mif_xy[n,1]),
                               'distance_px':float(dist[n]),'real_to_shift_ratio':float(row.real_to_shift_ratio)})
            print(json.dumps({'case':case,'patch':patch,'tier':tier,'candidates':len(labels),
                              'displayed':len(selected),'ratio':float(row.real_to_shift_ratio)}),flush=True)
    finally:
        for pair in slide_cache.values():
            for slide in pair:
                slide.close()
    legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor='black',
                   markersize=7,label=name) for c,name in zip(COLORS,CLASSES)]
    legend.append(Line2D([0],[0],marker='o',color='none',markerfacecolor='none',markeredgecolor='white',
                         markersize=7,label='matched H&E nucleus center'))
    fig.legend(handles=legend,ncol=3,loc='lower center',bbox_to_anchor=(.5,.005),fontsize=9,
               facecolor='#333333',labelcolor='white')
    fig.suptitle('CRC Panel1 registration review: same level-0 coordinates; colored = mIF center, white ring = H&E center',fontsize=13)
    fig.tight_layout(rect=(0,.045,1,.97),h_pad=1.5)
    fig.savefig(a.output/'three_case_registration_contact_sheet.png',dpi=170,facecolor='white')
    plt.close(fig)
    pd.DataFrame(detail).to_csv(a.output/'displayed_cells.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'cases':list(CASE_MAP),'patches':PICKS[:a.max_rows],
        'rows':a.max_rows,'displayed_cells':len(detail),
        'note':'DAPI display contrast stretched; original raw data unchanged. 2017 source label table was not re-read.'},indent=2)+'\n')


if __name__=='__main__':
    main()
