"""Join independently accepted diagnostics, with explicit sampling limitations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmarks.v5.encoder_research.continuous_data import sha


def describe(rows):
    present = [r for r in rows if r['metrics']['positive'] > 0]
    absent = [r for r in rows if r['metrics']['positive'] == 0]
    classes = defaultdict(list)
    for r in present:
        classes[r['plan']['dataset']+'|'+str(r['plan']['target'])].append(r['metrics']['dice'])
    def mean(values):
        values = [x for x in values if x is not None]
        return float(np.mean(values)) if values else None
    return dict(n=len(rows), present_n=len(present), absent_n=len(absent),
                mean_Dice=mean([r['metrics']['dice'] for r in present]),
                macro_Dice=mean([np.mean(x) for x in classes.values()]), class_n=len(classes),
                BF1=mean([r['metrics']['bf1_um2'] for r in present]),
                absent_FPA=mean([r['metrics']['false_positive_fraction'] for r in absent]),
                probability_std=mean([r['diagnostics']['std'] for r in rows]),
                swap_mean=mean([r['diagnostics']['swap_valid_mean'] for r in rows]),
                empty_present_fraction=mean([r['diagnostics']['empty_prediction'] for r in present]),
                support_present_Dice=mean([r['support_metrics']['dice'] for r in present]),
                per_class_Dice={k:float(np.mean(v)) for k,v in classes.items()})


def make_figures(root, rows, c):
    entries = {r['tile_id']:r for r in map(json.loads, (Path(c['cache'])/'index.jsonl').read_text().splitlines())}
    from benchmarks.v5.encoder_research.cell_catalog import load_cell_catalog
    from benchmarks.v5.encoder_research.continuous_data import TASKS
    _, inventory = load_cell_catalog(TASKS)
    selected = [(i,r) for i,r in enumerate(rows['I']) if r['split']=='val' and r['rank']==0]
    masks = {}
    for arm in ['I','S11']:
        with np.load(root/'full'/(arm+'_masks.npz')) as f:
            masks[arm] = f['masks'].copy()
    width, h = 256, 292
    canvas = Image.new('RGB', (5*width, 32+h*len(selected)), 'white')
    draw = ImageDraw.Draw(canvas)
    titles = ['Support: positive red / negative blue', 'Query image', 'Target GT (evaluation only)', 'Image-only prediction', 'S11 prediction']
    for col, title in enumerate(titles):
        draw.text((col*width+5, 10), title, fill='black')
    def image(identity):
        with np.load(entries[identity]['file']) as f:
            return Image.fromarray(f['image_0'].copy()).convert('RGB')
    def overlay(im, mask):
        arr = np.asarray(im).astype(float)
        arr[mask] = .5*arr[mask] + .5*np.array([0,255,0])
        return Image.fromarray(arr.astype(np.uint8))
    for rownum,(index,r) in enumerate(selected):
        plan = r['plan']; query = image(plan['query_id']); source = image(plan['support_id'])
        ds = ImageDraw.Draw(source)
        for sign,color in [('positive','red'),('negative','blue')]:
            for x,y in plan[sign+'_xy']:
                x*=512; y*=512; ds.ellipse((x-6,y-6,x+6,y+6),fill=color,outline='white',width=2)
        gt = np.asarray(Image.open(inventory[plan['query_id']]['labels']))
        panels = [source,query,overlay(query,gt==plan['target'])]
        for arm in ['I','S11']:
            mask=np.unpackbits(masks[arm][index],count=512*512).reshape(512,512).astype(bool)
            panels.append(overlay(query,mask))
        ypos = 32+rownum*h
        for col,im in enumerate(panels):
            canvas.paste(im.resize((width,width)),(col*width,ypos))
        draw.text((5,ypos+260),r['group']+' target '+str(plan['target']),fill='black')
        for col,arm in [(3,'I'),(4,'S11')]:
            m=rows[arm][index]['metrics']; text='absent FPA %.3f'%m['false_positive_fraction'] if m['dice'] is None else 'Dice %.3f'%m['dice']
            draw.text((col*width+5,ypos+260),text,fill='black')
    canvas.save(root/'fixed_examples.png')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args();root=a.root
    proof=json.loads((root/'full/acceptance.json').read_text())
    assert proof['status']=='accepted_frozen_checkpoint_diagnostic'
    c=json.loads((root/'contract.json').read_text())
    saved=json.loads((root/'saved_masks/report.json').read_text())
    result={}; rows={}
    for arm in c['arms']:
        rows[arm]=json.loads((root/'full'/(arm+'_rows.json')).read_text())
        groups={}
        for split in ['train','val']:
            for scope,kinds in [('local',['local']),('remote_present',['cross_present']),('remote_absent',['cross_absent'])]:
                subset=[r for r in rows[arm] if r['split']==split and r['plan']['kind'] in kinds]
                groups[split+'|'+scope]=describe(subset)
                for d in ['crc','tiger']:
                    groups[split+'|'+d+'|'+scope]=describe([r for r in subset if r['plan']['dataset']==d])
        gap={}
        for scope,kind in [('local','local'),('remote_present','cross_present')]:
            tr=groups['train|'+scope]['per_class_Dice'];va=groups['val|'+scope]['per_class_Dice']; common=sorted(tr.keys()&va.keys())
            gap[scope]=dict(common_classes=common,train_macro=float(np.mean([tr[k] for k in common])),
                            val_macro=float(np.mean([va[k] for k in common])),
                            train_minus_val=float(np.mean([tr[k]-va[k] for k in common])),
                            limitation='same class weighting, different images/area distribution; train prompts are consumed-task prefixes, no independent test')
        capacities=[r for r in rows[arm] if 'capacity' in r]
        representation=rows[arm][0]['representation']
        result[arm]=dict(groups=groups,class_matched_train_val=gap,capacity_train16=[dict(group=r['group'],target=r['plan']['target'],
                              positive_fraction=r['metrics']['positive']/r['metrics']['valid'],actual_Dice=r['metrics']['dice'],**r['capacity']) for r in capacities],
                        representation_shape_example=representation,
                        assignment_max_mean=float(np.mean([r['representation']['average_assignment_max'] for r in rows[arm]])),
                        argmax_used_slots_min=min(r['representation']['argmax_used_slots'] for r in rows[arm]),
                        probability_std_mean=float(np.mean([r['diagnostics']['std'] for r in rows[arm]])))
        paired=defaultdict(list)
        for record in saved['results'][arm]['same_query_target_pairs']:
            paired[(record['query_id'],record['target'])].append(record)
        result[arm]['matched_query_target']=dict(observations=sum(map(len,paired.values())),unique_pairs=len(paired),
            unique_queries=len({key[0] for key in paired}),
            local_Dice=float(np.mean([v[0]['local_Dice'] for v in paired.values()])),
            remote_Dice=float(np.mean([np.mean([r['remote_Dice'] for r in v]) for v in paired.values()])),
            remote_minus_local=float(np.mean([np.mean([r['difference'] for r in v]) for v in paired.values()])),
            limitation='remote supports averaged within each query-target first; repeated queries/patients remain dependent; task prompts differ')
        full=json.loads((root/'saved_masks'/(arm+'_rows.json')).read_text())
        for scope in ['local','remote']:
            subset=[r for r in full if r['scope']==scope and 0<r['positive_fraction']<.05]
            result[arm][scope+'_small_area_ratio_median']=float(np.median([r['pred_to_true_area'] for r in subset]))
    report=dict(status='accepted_basic_bottleneck_diagnostic_synthesis',results=result,
                full_saved_results=saved['results'],inference_acceptance_sha256=sha(root/'full/acceptance.json'),
                saved_report_sha256=sha(root/'saved_masks/report.json'),source_sha256=sha(__file__),
                source_contract_sha256=sha(root/'contract.json'),training_performed=False,
                limitations=['one trained seed; train192/val192 are diagnostic samples with different area distributions',
                             'training-fit deficiency is not proof that more steps or another loss will solve it',
                             'train-only least-squares projection is neither a deployable model nor a strict Dice upper bound',
                             '8um error band and existing paired local/remote tasks do not establish causal attribution',
                             '5+1 and 3+2 differ in both positive and negative clicks, not a pure click-count comparison'])
    (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    make_figures(root,rows,c)
    print(json.dumps(dict(status=report['status'],output=str(root))),flush=True)


if __name__=='__main__':
    main()
