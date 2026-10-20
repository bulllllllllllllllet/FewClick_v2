"""Audit all sources, freeze spatial tile manifests, and render standard inputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from .geometry import IGNORE, quality_version, rasterize_tissue, spatial_split, stable_bucket, standardize

BASE = Path('/nfs-medical3/zyh')
CRC = BASE/'fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235'
LABELS = {
    'crc': {i: x for i,x in enumerate(['tumor_epithelium','tumor_stroma','background','necrosis','normal_gland','normal_stroma','submucosa_serosa','muscle','lymphocyte_aggregate','mucus','fat','blood'])},
    'ocelot': {1:'noncancer_composite',2:'cancer_area'},
    'ignite': {1:'Background',2:'Tumor epithelium',3:'Reactive epithelium',4:'Stroma',5:'Inflammation',6:'Alveolar tissue',7:'Fatty tissue',8:'Necrotic tissue',9:'Erythrocytes',10:'Bronchial epithelium',11:'Mucus/Plasma/Fluids',12:'Cartilage/Bone',13:'Macrophages',14:'Muscle',15:'Liver',16:'Keratinization'},
    'tiger': {1:'invasive_tumor',2:'tumor_associated_stroma',3:'in_situ_tumor',4:'healthy_glands',5:'necrosis_not_in_situ',6:'inflamed_stroma',7:'rest_composite'},
    'puma': {1:'tissue_tumor',2:'tissue_stroma',3:'tissue_epidermis',4:'tissue_necrosis',5:'tissue_blood_vessel',6:'tissue_white_background'},
}
pyvips.cache_set_max(0)
Image.MAX_IMAGE_PIXELS = None


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def json_safe(row):
    return {k: (None if isinstance(v,float) and math.isnan(v) else v) for k,v in row.items()}


def read_rows(path):
    return [json_safe(r) for r in pd.read_parquet(path).to_dict('records')]


def tiff_mpp(path):
    with Image.open(path) as image:
        unit = int(image.tag_v2.get(296, 0))
        if unit not in (2,3):
            raise ValueError(f'untrusted TIFF resolution unit: {path}')
        factor = 25400 if unit == 2 else 10000
        mpp = [factor/float(image.tag_v2[k]) for k in (282,283)]
        size = image.size
    if not all(math.isfinite(v) and .05 < v < 5 for v in mpp):
        raise ValueError(f'invalid MPP: {path} {mpp}')
    vips = pyvips.Image.new_from_file(str(path), access='random')
    if max(abs(mpp[i]-1000/value) for i,value in enumerate((vips.xres,vips.yres))) > 1e-5:
        raise ValueError(f'TIFF and vips scale disagree: {path}')
    return mpp, size


@lru_cache(maxsize=None)
def file_stat(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.stat()


def base_source(dataset, identity, image, annotation, mpp, size, split, slide=None, patient=None, organ='unknown', **extra):
    image, annotation = Path(image), Path(annotation)
    stat = file_stat(image)
    return dict(source_id=f'{dataset}:{identity}', dataset=dataset, organ=organ,
        disease={'crc':'colorectal_cancer','ocelot':'six_organ_cohort','ignite':'NSCLC','tiger':'breast_cancer','puma':'melanoma'}[dataset],
        image=str(image), annotation=str(annotation), available_mpp_x=float(mpp[0]), available_mpp_y=float(mpp[1]),
        width=int(size[0]), height=int(size[1]), split=split, official_split=extra.pop('official_split',split),
        slide_id=slide, slide_id_verified=slide is not None, patient_id=patient,
        training_allowed=dataset != 'puma' and split == 'train',
        image_size_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns,
        annotation_size_bytes=file_stat(annotation).st_size, extra_json=json.dumps(extra,ensure_ascii=False))


def source_catalog():
    rows, notes = [], []
    patches = pd.read_parquet(CRC/'patches_index.parquet')
    spatial = pd.read_parquet(CRC/'spatial_manifest.parquet').set_index('patch_id')
    headers = {p:tiff_mpp(p)[0] for p in sorted(patches.wsi_path.unique())}
    for r in patches.to_dict('records'):
        x = spatial.loc[r['patch_id']]
        rows.append(base_source('crc',r['patch_id'],r['wsi_path'],r['gt_path'],headers[r['wsi_path']],
            [r['width_level0'],r['height_level0']],r['split'],r['wsi_id'],r['patient_id'],'colorectum',
            version='existing_spatial_0922_001235',spatial_group_id=x.spatial_group_id,
            image_window=[r[k] for k in ('x_level0','y_level0','width_level0','height_level0')],
            gt_window=[r[k] for k in ('x_10x','y_10x','width_10x','height_10x')],
            historical_split=r['historical_split'], initialization_exposure='historical encoder exposure; not blind'))
    root = BASE/'OCELOT_1016_122128/extracted/ocelot2023_v1.0.1'
    metadata = json.loads((root/'metadata.json').read_text())['sample_pairs']
    for identity, r in sorted(metadata.items()):
        subset = r['subset']
        rows.append(base_source('ocelot',identity,root/f'images/{subset}/tissue/{identity}.jpg',root/f'annotations/{subset}/tissue/{identity}.png',
            [r['tissue'][f'resized_mpp_{a}'] for a in ('x','y')],[1024,1024],subset,r['slide_name'],r['slide_name'].split('-01Z')[0],r['organ'],
            version='1.0.1', scanner_mpp_xy=[r['mpp_x'],r['mpp_y']], metadata=r,
            paired_high_detail_image=str(root/f'images/{subset}/cell/{identity}.jpg')))
    root = BASE/'IGNITE_1016_122128'
    for r in pd.read_csv(root/'data_overview.csv').fillna('').to_dict('records'):
        if r['task'] != 'he_tissue_segmentation':
            continue
        split = 'test' if r['split'] == 'test' else 'val' if r['validation_fold'] == 'fold0' else 'train'
        image = root/f'extracted/images/he/{r["name"]}.png'
        with Image.open(image) as im:
            size = im.size
        rows.append(base_source('ignite',r['name'],image,root/f'extracted/annotations/he/{r["name"]}.png',[.5,.5],size,split,
            patient=str(r['patient_id']),organ=r['organ'],version='v2_17735903',official_split=r['split'],
            scanner=r['scanner'],histological_subtype=r['histological_subtype'],validation_fold=r['validation_fold'],
            mpp_evidence='official released 0.5MPP; area/shape metadata previously cross-checked',context_used=False))
    root = BASE/'TIGER_1016_122128/roi_reuse_1016_122128/wsirois/roi-level-annotations'
    pattern = re.compile(r'^(.*)_\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]$')
    for group in sorted(root.iterdir()):
        for image in sorted((group/'images').glob('*.png')):
            annotation = group/'masks'/image.name
            if not annotation.exists():
                notes.append(dict(dataset='tiger',image=str(image),reason='published_image_without_mask'));continue
            match = pattern.fullmatch(image.stem)
            if not match:
                raise ValueError(f'TIGER coordinate filename: {image}')
            slide = match[1]; x0,y0,x1,y1 = map(int,match.groups()[1:])
            with Image.open(image) as im:
                size = im.size
            if size != (x1-x0,y1-y0):
                notes.append(dict(dataset='tiger',image=str(image),reason='quarantined_coordinate_grid_mismatch',
                    released_bbox=[x0,y0,x1,y1],actual_size=list(size)))
                continue
            rows.append(base_source('tiger',f'{group.name}:{image.stem}',image,annotation,[.5,.5],size,'spatial_pending',slide,organ='breast',
                version='AWS_WSIROIS_1016_122128',source_group=group.name,slide_origin_xy=[x0,y0],
                mpp_evidence='official release ~0.5MPP; coordinates in published grid'))
    root = BASE/'PUMA_1016_122128'
    for image in sorted((root/'extracted/01_training_dataset_tif_ROIs').glob('*.tif')):
        mpp,size = tiff_mpp(image)
        bucket = stable_bucket('puma|'+image.stem)
        split = 'train' if bucket < 70 else 'val' if bucket < 85 else 'test'
        rows.append(base_source('puma',image.stem,image,root/f'extracted/01_training_dataset_geojson_tissue/{image.stem}_tissue.geojson',mpp,size,split,
            organ='melanoma_site_unreported',version='v5_15050523',official_split='public_training',
            holdout_domain=True,specimen_group='metastatic' if 'metastatic' in image.stem else 'primary'))
    notes.extend([dict(dataset='puma',reason='orphan_annotation_excluded',identity='training_set_metastatic_roi_103'),
        dict(dataset='tiger',reason='whole_WSI_download_failed; no complete WSI images; ROI inputs only')])
    ids = [r['source_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source IDs')
    # Official source slide footprints must not cross splits for OCELOT.
    groups = defaultdict(set)
    for r in rows:
        if r['dataset'] == 'ocelot':
            groups[r['slide_id']].add(r['split'])
    if any(len(s)>1 for s in groups.values()):
        raise ValueError('OCELOT official slide leakage')
    return rows, notes


def tile_catalog(sources):
    tiles, notes, duplicates = {}, Counter(), []
    for source in sources:
        e = json.loads(source['extra_json']); dataset = source['dataset']
        w,h = (max(1,round(256/source[k])) for k in ('available_mpp_x','available_mpp_y'))
        width,height = source['width'],source['height']
        if dataset == 'tiger':
            sx,sy = e['slide_origin_xy']
            xs = list(range(math.ceil(sx/w)*w-sx, width-w+1,w))
            ys = list(range(math.ceil(sy/h)*h-sy, height-h+1,h))
        elif dataset in ('crc','puma'):
            xs,ys = [(width-w)//2],[(height-h)//2]
        else:
            xs = list(range(0,width-w+1,w)) if width>=w else [(width-w)//2]
            ys = list(range(0,height-h+1,h)) if height>=h else [(height-h)//2]
        if not xs or not ys:
            notes[f'{dataset}:no_complete_global_grid_tile'] += 1
        for y in ys:
            for x in xs:
                split = source['split'];group = e.get('spatial_group_id',source['source_id'])
                if dataset == 'tiger':
                    assignment = spatial_split(source['slide_id'],[sx+x,sy+y,w,h])
                    if assignment is None:
                        notes['tiger:buffered_block_boundary_tile'] += 1;continue
                    split,group = assignment['split'],assignment['spatial_group_id']
                    key = f'tiger|{source["slide_id"]}|{sx+x}|{sy+y}|{w}|{h}'
                else:
                    key = f'{source["source_id"]}|{x}|{y}|{w}|{h}'
                r = dict(tile_id=sha(key.encode())[:24],source_id=source['source_id'],dataset=dataset,organ=source['organ'],
                    split=split,spatial_group_id=group,training_allowed=split=='train' and dataset!='puma',
                    image_box_json=json.dumps([x,y,w,h]),selection_hash=sha(('selection|'+key).encode()),
                    slide_id=source['slide_id'],slide_id_verified=source['slide_id_verified'])
                if key in tiles:
                    previous = tiles[key]
                    prefer = dataset == 'tiger' and e['source_group']=='tissue-cells'
                    duplicates.append(dict(tile_id=r['tile_id'],kept_source=r['source_id'] if prefer else previous['source_id'],
                        excluded_source=previous['source_id'] if prefer else r['source_id'],reason='identical_released_WSI_footprint'))
                    if not prefer:
                        continue
                tiles[key] = r
    rows = sorted(tiles.values(),key=lambda r:r['tile_id'])
    groups = defaultdict(set)
    for row in rows:
        groups[row['spatial_group_id']].add(row['split'])
    if any(len(v)>1 for v in groups.values()):
        raise ValueError('spatial group crosses split')
    return rows,dict(notes),duplicates


def native_inputs(source):
    e = json.loads(source['extra_json'])
    if source['dataset'] == 'crc':
        image = pyvips.Image.new_from_file(source['image'],access='random').crop(*e['image_window']).extract_band(0,n=3)
        rgb = np.frombuffer(image.write_to_memory(),np.uint8).reshape(image.height,image.width,3).copy()
        from benchmarks.v4.phase_1_multiscale.src.data import decode_gt_patch
        config = yaml.safe_load(Path('benchmarks/v4/phase_2_region_encoder/configs/phase2_region_10x.yaml').read_text())
        labels = decode_gt_patch(Path(source['annotation']),*e['gt_window'],config['data']['class_map'],IGNORE)
        audit = {'label_grid_differs_from_RGB': True}
    else:
        with Image.open(source['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        if source['dataset'] == 'puma':
            labels,audit = rasterize_tissue(json.loads(Path(source['annotation']).read_text()),rgb.shape[:2],{v:k for k,v in LABELS['puma'].items()})
        else:
            with Image.open(source['annotation']) as im:
                if im.size != (rgb.shape[1],rgb.shape[0]):
                    raise ValueError('source GT/image extent mismatch')
                labels = np.asarray(im).copy()
            if labels.ndim != 2:
                raise ValueError('GT must be single channel')
            allowed = set(LABELS[source['dataset']]) | ({255} if source['dataset']=='ocelot' else {0})
            if set(np.unique(labels))-allowed:
                raise ValueError('unknown source labels')
            if source['dataset'] in ('ignite','tiger'):
                labels[labels==0] = IGNORE
            audit = {}
    if set(np.unique(labels)) - set(LABELS[source['dataset']]) - {IGNORE}:
        raise ValueError('decoded label schema mismatch')
    return rgb,labels,audit


def make_catalog(output):
    output.mkdir(parents=True,exist_ok=False)
    start = time.time()
    try:
        sources,notes = source_catalog()
        tiles,skip,duplicates = tile_catalog(sources)
        pd.DataFrame(sources).to_parquet(output/'sources.parquet',index=False)
        pd.DataFrame(tiles).to_parquet(output/'tiles.parquet',index=False)
        (output/'deduplicated.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in duplicates))
        counts = Counter((r['dataset'],r['split']) for r in tiles)
        report = dict(status='complete',worker_pid=os.getpid(),source_counts=dict(Counter(r['dataset'] for r in sources)),
            tile_counts=[dict(dataset=d,split=s,tiles=n) for (d,s),n in sorted(counts.items())],
            labels=LABELS,ignore_index=IGNORE,source_notes=notes,tile_exclusions=skip,deduplicated_footprints=len(duplicates),
            target_mpp=.5,target_extent_um=256,input_size=512,seed=20261020,case_grouping=False,
            puma_training_allowed=False,test_pixels_read=False,
            tigER_split='0.5MPP published coordinate grid; 8192px blocks,64px margins; 70/15/15 by block',
            overlap_audit='TIGER canonical global grid and buffered blocks; CRC original buffered context split; OCELOT official slides; other sources complete ROI groups',
            initial_encoder_exposure='historical CRC; external pretraining provenance not guaranteed unseen',
            seconds=time.time()-start,source_sha256={str(p):sha(p.read_bytes()) for p in [Path(__file__),Path(__file__).with_name('geometry.py')]})
        dump(output/'summary.json',report)
        dump(output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-start))
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


def render(catalog,output,per_group,qualities=('original','0.8','1.0')):
    output.mkdir(parents=True,exist_ok=False)
    (output/'patches').mkdir()
    start = time.time();sources = {r['source_id']:r for r in read_rows(catalog/'sources.parquet')}
    tiles = sorted(read_rows(catalog/'tiles.parquet'),key=lambda r:r['selection_hash'])
    counts,skips = Counter(),Counter();rows=[];image_splits=defaultdict(set)
    try:
        # Selection is score-independent; only train/val pixels, including declared PUMA holdout.
        with (output/'progress.jsonl').open('w') as log:
            for tile in tiles:
                if tile['split']=='test':
                    continue
                key = (tile['dataset'],tile['organ'],tile['split'])
                if counts[key] >= per_group:
                    continue
                source = sources[tile['source_id']]
                rgb,labels,audit = native_inputs(source)
                image,gt,meta = standardize(rgb,labels,json.loads(tile['image_box_json']),[source['available_mpp_x'],source['available_mpp_y']])
                if (gt!=IGNORE).mean()<.25:
                    skips[f'{tile["dataset"]}:valid_fraction_below_0.25']+=1;continue
                folder = output/'patches'/tile['tile_id'];folder.mkdir()
                tile_rows = []
                Image.fromarray(gt).save(folder/'labels.png')
                label_hash = sha(gt.tobytes())
                gt_counts = {int(k):int(v) for k,v in zip(*np.unique(gt,return_counts=True))}
                for quality in qualities:
                    value,qmeta = quality_version(image,meta['input_mpp_xy'],[source['available_mpp_x'],source['available_mpp_y']],quality,meta)
                    path = folder/f'image_{quality}.png'
                    Image.fromarray(value).save(path,compress_level=1)
                    raw_hash = sha(value.tobytes());image_splits[raw_hash].add(tile['split'])
                    row = dict(tile,quality=quality,image=str(path),labels=str(folder/'labels.png'),input_mpp_xy_json=json.dumps(meta['input_mpp_xy']),
                        available_mpp_xy_json=json.dumps([source['available_mpp_x'],source['available_mpp_y']]),
                        image_sha256=sha(path.read_bytes()),rgb_sha256=raw_hash,labels_array_sha256=label_hash,
                        class_counts_json=json.dumps(gt_counts),valid_fraction=float((gt!=IGNORE).mean()),
                        cell_features_status='not_prepared',preparation_json=json.dumps(dict(meta,**qmeta,**audit)))
                    rows.append(row)
                    tile_rows.append(row)
                    with Image.open(path) as im:
                        if not np.array_equal(np.asarray(im),value):
                            raise ValueError('saved RGB mismatch')
                with Image.open(folder/'labels.png') as im:
                    if not np.array_equal(np.asarray(im),gt):
                        raise ValueError('saved GT mismatch')
                counts[key]+=1
                dump(folder/'details.json',tile_rows)
                log.write(json.dumps(dict(tile_id=tile['tile_id'],dataset=tile['dataset'],organ=tile['organ'],split=tile['split'],variants=len(qualities)))+'\n');log.flush()
                if sum(counts.values())%20==0:
                    dump(output/'status.json',dict(status='running',worker_pid=os.getpid(),completed_tiles=sum(counts.values()),
                        input_variants=len(rows),seconds=time.time()-start))
                    print('RENDERED',sum(counts.values()),'seconds',round(time.time()-start,1),flush=True)
        # Exact uniform images can coincide naturally; report all duplicate contents,
        # but do not claim content identity alone establishes shared physical origin.
        content_collisions = {k:sorted(v) for k,v in image_splits.items() if len(v)>1}
        pd.DataFrame(rows).to_parquet(output/'inputs.parquet',index=False)
        report = dict(status='complete',worker_pid=os.getpid(),catalog=str(catalog),per_dataset_organ_split_cap=per_group,
            source_manifest_sha256=sha((catalog/'sources.parquet').read_bytes()),tile_manifest_sha256=sha((catalog/'tiles.parquet').read_bytes()),
            source_sha256={str(p):sha(p.read_bytes()) for p in [Path(__file__),Path(__file__).with_name('geometry.py')]},
            selected_tiles=sum(counts.values()),input_variants=len(rows),qualities=list(qualities),
            counts=[dict(dataset=d,organ=o,split=s,tiles=n) for (d,o,s),n in sorted(counts.items())],
            skips=dict(skips),cross_split_equal_RGB_contents=content_collisions,
            test_pixels_read=False,puma_training_allowed=False,cell_features_status='not_prepared',seconds=time.time()-start)
        dump(output/'summary.json',report);dump(output/'status.json',dict(status='complete',worker_pid=os.getpid(),seconds=time.time()-start))
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(output/'status.json',dict(status='failed',worker_pid=os.getpid(),error=repr(exc)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['catalog','render'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path)
    p.add_argument('--per-group',type=int,default=16)
    a=p.parse_args()
    if a.per_group<1:
        raise ValueError('positive sample cap required')
    if a.mode=='catalog':make_catalog(a.output)
    else:
        if a.catalog is None:raise ValueError('--catalog required')
        render(a.catalog,a.output,a.per_group)


if __name__=='__main__':main()
