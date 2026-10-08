#!/usr/bin/env python3
"""Show the same H&E cell at 16/32/64/128 px in one contact sheet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from benchmarks.v5.tools.run_crc_panel1_cell_identity import CLASSES, SOURCE


SUP=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3/panel1_eval_supplement_0923_155834')
ROOT=Path('/nfs-medical3/zyh/fewclick_v2')
SIZES=(16,32,64,128)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def crop(array,cx,cy,size):
    h=size//2
    return array[cy-h:cy+h,cx-h:cx+h]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT):
        p.error('output must be in FewClick artifact root')
    a.output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(SUP/'cells.csv')
    meta=json.loads((SUP/'audit.json').read_text())
    if digest(SUP/'cells.csv')!=meta['manifest_sha256']:
        raise ValueError('supplement manifest hash mismatch')
    frame=frame.loc[frame.split.eq('val')].copy()
    nuclei={};images={};rows=[]
    for patch,group in frame.groupby('patch',sort=False):
        image_path=SOURCE/'registered_he_patches'/f'{patch}.png'
        mask_path=SOURCE/'cell_matching/masks'/f'{patch}.npy'
        if digest(image_path)!=group.he_sha256.iloc[0] or digest(mask_path)!=group.mask_sha256.iloc[0]:
            raise ValueError(f'raw source hash mismatch: {patch}')
        image=np.asarray(Image.open(image_path).convert('RGB'))
        mask=np.load(mask_path,allow_pickle=False)
        if image.shape!=(512,512,3) or mask.shape!=(512,512):
            raise ValueError('source image/mask size mismatch')
        images[patch]=image
        for rec in group.itertuples():
            cx,cy=round(float(rec.he_x)),round(float(rec.he_y))
            target=mask==int(rec.he_id)
            if not target.any() or not 64<=cx<=447 or not 64<=cy<=447:
                raise ValueError(f'target nucleus missing or unsafe: {patch}/{rec.he_id}')
            key=(patch,int(rec.he_id))
            nuclei[key]=target
            item={'patch':patch,'he_id':int(rec.he_id),'mif_id':int(rec.mif_id),
                  'label':int(rec.label),'he_x':float(rec.he_x),'he_y':float(rec.he_y),
                  'nucleus_area_px':int(target.sum())}
            for size in SIZES:
                item[f'fraction_{size}']=float(crop(target,cx,cy,size).sum()/target.sum())
            rows.append(item)
    data=pd.DataFrame(rows)
    chosen=[]
    for klass in range(5):
        values=data.loc[data.label.eq(klass)].copy()
        median=values.nucleus_area_px.median()
        chosen.append(values.iloc[np.argmin(abs(values.nucleus_area_px-median))])
    used={(r['patch'],int(r['he_id'])) for r in chosen}
    truncated=data.loc[~data.apply(lambda r:(r.patch,int(r.he_id)) in used,axis=1)].sort_values('fraction_16').iloc[0]
    chosen.append(truncated)
    fig,axes=plt.subplots(len(chosen),4,figsize=(12,2.75*len(chosen)),squeeze=False)
    details=[]
    for i,row in enumerate(chosen):
        patch=row['patch'];cx,cy=round(float(row['he_x'])),round(float(row['he_y']))
        target=nuclei[(patch,int(row['he_id']))]
        label='large-nucleus example' if i==5 else CLASSES[int(row['label'])]
        for j,size in enumerate(SIZES):
            raw=crop(images[patch],cx,cy,size)
            mask=crop(target,cx,cy,size)
            if raw.shape!=(size,size,3) or mask.shape!=(size,size):
                raise ValueError('invalid visualization crop')
            shown=np.asarray(Image.fromarray(raw).resize((224,224),Image.Resampling.BILINEAR)).copy()
            boundary=mask & ~ndimage.binary_erosion(mask)
            outline=np.asarray(Image.fromarray(boundary.astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST))>0
            shown[outline]=np.array([255,245,0],np.uint8)
            ax=axes[i,j];ax.imshow(shown,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{size}×{size} px · nucleus {row[f"fraction_{size}"]:.0%}',fontsize=9)
        axes[i,0].set_ylabel(f'{label}\n{patch} / HE {int(row["he_id"])}',fontsize=9)
        details.append({**row.to_dict(),'display_role':label})
    fig.suptitle('Same H&E nucleus at four crop sizes (all resized to 224 px for display); yellow = target nucleus outline',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97),h_pad=1.1)
    fig.savefig(a.output/'crop_size_contact_sheet.png',dpi=170,facecolor='white')
    plt.close(fig)
    pd.DataFrame(details).to_csv(a.output/'displayed_cells.csv',index=False)
    (a.output/'manifest.json').write_text(json.dumps({'source_manifest_sha256':meta['manifest_sha256'],
        'rows':len(chosen),'sizes':SIZES,'note':'raw H&E shown; model additionally applies nucleus soft attention before resizing'},indent=2)+'\n')
    print(json.dumps({'status':'complete','output':str(a.output),'rows':len(chosen)}),flush=True)


if __name__=='__main__':
    main()
