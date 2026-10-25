"""Audit available physical context, then render one fixed centre per eligible ROI."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.v5.fair_comparison.protocol import resolve_clicks, nested
from benchmarks.v5.multidomain_data.prepare import native_inputs, LABELS
from benchmarks.v5.multidomain_training.catalog import interior_labels, TARGETS
from benchmarks.v5.paper_multiscale.data import context_geometry, native_context

SOURCES = Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/prepare_1020_174937/catalog_1020_180034/sources.parquet')
TRAINING = Path('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(output, sources_path=SOURCES, training_path=TRAINING):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'status.json', dict(status='preparing', worker_pid=os.getpid()))
    try:
        sources = pd.read_parquet(sources_path).to_dict('records')
        training = pd.read_parquet(training_path).to_dict('records')
        if any(r['dataset'] in ('ignite', 'puma', 'ocelot') for r in training):
            raise ValueError('external domain occurs in current development catalog')
        candidates = [r for r in sources if r['dataset'] in ('ignite', 'puma', 'ocelot')]
        audit, selected = [], []
        for source in sorted(candidates, key=lambda r: r['source_id']):
            mx, my = source['available_mpp_x'], source['available_mpp_y']
            if not np.isfinite([mx, my]).all() or min(mx, my) <= 0:
                raise ValueError('untrusted MPP')
            cx, cy = source['width'] * mx / 2, source['height'] * my / 2
            row = dict(source_id=source['source_id'], dataset=source['dataset'],
                       x0_um=cx-128, x1_um=cx+128, y0_um=cy-128, y1_um=cy+128)
            try:
                geometry = context_geometry(row, source)
                status = 'eligible'
            except ValueError as exc:
                if str(exc) != 'incomplete_native_context':
                    raise
                status = str(exc)
            audit.append(dict(source_id=source['source_id'], dataset=source['dataset'],
                original_split=source['split'], width=source['width'], height=source['height'],
                mpp_x=mx, mpp_y=my, width_um=source['width']*mx,
                height_um=source['height']*my, status=status))
            if status == 'eligible':
                if source['dataset'] != 'ignite':
                    raise ValueError('unexpected eligible domain; define its rendering contract explicitly')
                selected.append((source, row, geometry))
        pd.DataFrame(audit).to_parquet(output/'availability.parquet', index=False)
        if not selected:
            raise ValueError('no native full-context field')
        rows, tasks = [], []
        for source, row, geometry in selected:
            rgb, labels, _ = native_inputs(source)
            if rgb.shape[:2] != (source['height'], source['width']):
                raise ValueError('actual image header differs from catalog')
            if labels.shape != rgb.shape[:2]:
                raise ValueError('native label grid differs from RGB')
            tile_id = hashlib.sha256(('external-centre-v1|'+source['source_id']).encode()).hexdigest()[:24]
            folder = output/tile_id
            folder.mkdir()
            geometry_row = row | dict(multiscale_json=json.dumps(geometry))
            images = []
            for scale, box in enumerate(geometry['native_boxes']):
                x, y, w, h = box
                native = rgb[y:y+h, x:x+w]
                if native.shape != (h, w, 3) or not np.array_equal(native, native_context(geometry_row, scale)):
                    raise ValueError('native crop replay differs')
                image = np.asarray(Image.fromarray(native).resize((512,512), Image.Resampling.BILINEAR))
                path = folder/f'image_{scale}.png'
                Image.fromarray(image).save(path)
                if not np.array_equal(np.asarray(Image.open(path)), image):
                    raise ValueError('image reload differs')
                images.append(dict(path=str(path), sha256=digest(path)))
            x,y,w,h = geometry['native_boxes'][0]
            gt = np.asarray(Image.fromarray(labels[y:y+h,x:x+w]).resize((512,512), Image.Resampling.NEAREST)).copy()
            if set(np.unique(gt))-set(LABELS['ignite'])-{255}:
                raise ValueError('label ontology mismatch')
            gt_path = folder/'labels.png'
            Image.fromarray(gt).save(gt_path)
            if not np.array_equal(np.asarray(Image.open(gt_path)), gt):
                raise ValueError('GT reload differs')
            mpp = [w*geometry['native_mpp'][0]/512, h*geometry['native_mpp'][1]/512]
            inside = interior_labels(gt, mpp)
            counts = Counter(map(int,gt.ravel()))
            inner = Counter(map(int,inside.ravel()))
            valid, total = gt.size-counts[255], inside.size-inner[255]
            eligible = [c for c in TARGETS['ignite'] if counts[c]>=64 and valid-counts[c]>=64
                        and inner[c]>=5 and total-inner[c]>=5]
            row |= dict(tile_id=tile_id, original_split=source['split'], split='external',
                labels=str(gt_path), labels_array_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
                input_mpp_xy_json=json.dumps(mpp), multiscale_json=json.dumps(geometry),
                images_json=json.dumps(images), eligible_json=json.dumps(eligible),
                counts_json=json.dumps(dict(counts)), native_image_sha256=digest(source['image']),
                native_annotation_sha256=digest(source['annotation']), training_allowed=False,
                cell_cache_status='not_prepared', organ=source['organ'])
            rows.append(row)
            for target in eligible:
                pair_id = 60000000 + len({t['pair_id'] for t in tasks})
                for group in range(5):
                    plan = dict(pair_id=pair_id, number=pair_id, dataset='ignite', target=target,
                        support_id=tile_id, query_id=tile_id, prompt_group=group,
                        click_seed=20261025+pair_id+group*1000000000, kind='local',
                        protocol='local', split='external', gt_simulated_clicks=True)
                    plan = resolve_clicks(plan,row)
                    for name, sign in [('positive_xy',True),('negative_xy',False)]:
                        for xx,yy in plan[name]:
                            ix,iy = min(511,int(xx*512)),min(511,int(yy*512))
                            if gt[iy,ix] == 255 or bool(gt[iy,ix] == target) != sign:
                                raise ValueError('point sign/coordinate mismatch')
                    for budget in ('3+2','1+1','3+1','5+1'):
                        nested(plan,budget)
                    tasks.append(plan)
        pd.DataFrame(rows).to_parquet(output/'fields.parquet',index=False)
        write(output/'local_maximum.json',tasks)
        write(output/'contract.json',dict(status='prepared_inputs_and_clicks',worker_pid=os.getpid(),
            version='external_native_centre_v1', source_catalog_sha256=digest(sources_path),
            current_training_catalog_sha256=digest(training_path),
            source_sha256={str(Path(__file__)):digest(__file__)},
            input_fields_um=[256,512,1024], output_size=512, centre_rule='unique geometric ROI centre; no GT-based recentering',
            available_counts=dict(Counter(r['dataset']+'|'+r['status'] for r in audit)),
            selected_fields=len(rows), original_splits=dict(Counter(r['original_split'] for r in rows)),
            tasks=len(tasks), pairs=len({p['pair_id'] for p in tasks}),
            fields_with_targets=sum(bool(json.loads(r['eligible_json'])) for r in rows),
            ontology=LABELS['ignite'], maximum_points=[5,2], report_budgets=['3+2','1+1','3+1','5+1'],
            fields_sha256=digest(output/'fields.parquet'), clicks_sha256=digest(output/'local_maximum.json'),
            scope='External to current CRC/TIGER adaptation; not proof of unseen upstream image pretraining or independent patients',
            model_selection='Use CRC/TIGER selected weights and frozen thresholds; no external fitting or model selection',
            pending=['native cell caches','real model input gate','all matching baseline scores','source-cluster statistics'],
            provenance='XCell alignment trained on internal data per user; distinct upstream public priors'))
        write(output/'status.json',dict(status='complete_preparation', worker_pid=os.getpid(), fields=len(rows),tasks=len(tasks)))
    except Exception as exc:
        write(output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    build(args.output)
