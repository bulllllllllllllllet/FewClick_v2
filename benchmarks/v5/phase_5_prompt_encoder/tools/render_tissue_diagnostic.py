"""Render saved H&E, GT, predictions and identical support locations."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--timestamp', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}_\d{6}', args.timestamp): raise ValueError('timestamp required')
    manifest = json.loads((args.run/'manifest.json').read_text())
    # Fixed examples by ontology, never chosen by prediction performance.
    episodes = [e for e in manifest['episodes'] if e['target_class'] in (0,7,8) and e['prompt_size']=='small']
    figure, axes = plt.subplots(len(episodes), 5, figsize=(15, 3*len(episodes)), squeeze=False)
    for row, ep in enumerate(episodes):
        prefix = args.run / ('episode_'+ep['episode_id'])
        rgb = np.asarray(Image.open(str(prefix)+'_he.png'))
        with np.load(str(prefix)+'.npz') as masks:
            h, w = masks['gt'].shape
            for col, key in enumerate(('he', 'gt', 'phase5', 'single', 'multi')):
                axis = axes[row,col]
                if key=='he': axis.imshow(rgb)
                else: axis.imshow(np.ma.masked_where(~masks['valid'], masks[key]), cmap='gray', vmin=0, vmax=1)
                for sign, color, marker in [('positive', 'lime', '+'), ('negative', 'red', 'x')]:
                    xy = np.asarray(ep[sign+'_xy'])
                    axis.scatter(xy[:,0]*w,xy[:,1]*h,c=color,marker=marker,s=35)
                name = manifest['task_definitions'][str(ep['target_class'])]['task_id']
                axis.set_title(name.removeprefix('tissue.')+' / '+key,fontsize=9)
                axis.set_xlim(-.5,w-.5);axis.set_ylim(h-.5,-.5);axis.axis('off')
    figure.suptitle('Same-extent region-support diagnostic; green + / red x: support centroids',fontsize=12)
    figure.tight_layout()
    path = args.run/('contact_'+args.timestamp+'.png')
    if path.exists(): raise FileExistsError(path)
    figure.savefig(path,dpi=150);plt.close(figure)
    print(path)

if __name__ == '__main__': main()
