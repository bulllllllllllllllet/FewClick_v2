"""Reopen prepared categorical GT/RGB and expose a fixed data QA overview."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .prepare import LABELS,read_rows,dump,sha


def validate(inputs, output):
    output.mkdir(parents=True,exist_ok=False)
    rows=read_rows(inputs);groups=defaultdict(list)
    for row in rows:
        if row['split']=='test':raise ValueError('initial preparation must not decode test')
        if row['dataset']=='puma' and row['training_allowed']:raise ValueError('holdout domain enabled for training')
        image=np.asarray(Image.open(row['image']).convert('RGB'))
        gt=np.asarray(Image.open(row['labels']))
        if image.shape!=(512,512,3) or gt.shape!=(512,512):raise ValueError('standard input shape')
        if sha(Path(row['image']).read_bytes())!=row['image_sha256'] or sha(image.tobytes())!=row['rgb_sha256']:
            raise ValueError('image identity')
        if sha(gt.tobytes())!=row['labels_array_sha256']:raise ValueError('label identity')
        counts={str(int(k)):int(v) for k,v in zip(*np.unique(gt,return_counts=True))}
        if counts!=json.loads(row['class_counts_json']):raise ValueError('GT counts')
        if set(map(int,counts))-set(LABELS[row['dataset']])-{255}:raise ValueError('label ontology')
        meta=json.loads(row['preparation_json']);mpp=json.loads(row['input_mpp_xy_json'])
        if max(abs(x-.5) for x in mpp)>.002:raise ValueError('physical scale')
        affine=meta['input_to_image_affine'];sw,sh=meta['source_image_size']
        x=(np.arange(512)+.5)*affine[0]+affine[2];y=(np.arange(512)+.5)*affine[4]+affine[5]
        valid=(x[None,:]>=0)&(x[None,:]<sw)&(y[:,None]>=0)&(y[:,None]<sh)
        if (gt[~valid]!=255).any() or (image[~valid]!=255).any():raise ValueError('padding semantics')
        groups[row['tile_id']].append(row)
    for group in groups.values():
        if {r['quality'] for r in group}!={'original','0.8','1.0'}:raise ValueError('missing quality variant')
        if len({r['labels_array_sha256'] for r in group})!=1:raise ValueError('quality changed GT')
        if len({r['spatial_group_id'] for r in group})!=1 or len({r['split'] for r in group})!=1:raise ValueError('quality split leakage')
    # Prediction-independent: first selection hash per dataset, regardless of appearance.
    examples=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        first=min((r for r in rows if r['dataset']==dataset),key=lambda r:r['selection_hash'])
        examples.append({r['quality']:r for r in groups[first['tile_id']]})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(examples),4,figsize=(13,3.3*len(examples)),squeeze=False)
    colors=plt.get_cmap('tab20')
    for axes_row,example in zip(axes,examples):
        original=example['original'];gt=np.asarray(Image.open(original['labels']))
        for ax,quality in zip(axes_row,['original','original','0.8','1.0']):
            ax.imshow(Image.open(example[quality]['image']));ax.axis('off')
            ax.set_title(original['dataset']+' | '+quality,fontsize=10)
        overlay=np.zeros((*gt.shape,4),np.float32)
        for label in np.unique(gt):
            if label!=255:overlay[gt==label]=[*colors(int(label)%20)[:3],.35]
        axes_row[1].imshow(overlay)
        names=[LABELS[original['dataset']][int(k)] for k in sorted(np.unique(gt)) if k!=255]
        axes_row[1].set_title('GT: '+' / '.join(names),fontsize=8,wrap=True)
    fig.tight_layout()
    figure=output/'qa_overview_1020_174937.png';fig.savefig(figure,dpi=110,bbox_inches='tight');plt.close(fig)
    report=dict(status='passed',inputs=len(rows),unique_tiles=len(groups),datasets=dict(Counter(r['dataset'] for r in rows)),
        checks=['RGB/GT shapes and content hashes','label-code counts','physical scale','padding ignore/white',
                'same GT across quality variants','quality spatial group/split identity','PUMA holdout','test not decoded'],
        qa_overview=str(figure),cells_complete=sum(r['cell_features_status']=='complete' for r in rows))
    dump(output/'validation.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.inputs,a.output)


if __name__=='__main__':main()
