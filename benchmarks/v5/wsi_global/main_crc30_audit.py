"""Recount original CRC test30 RGB annotations and the physical valid4x domain."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import pyvips
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from .plan import geometry, grid

COHORT = Path('/nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet')
TILED = Path('/nfs-medical3/zyh/v4/phase1/data/tiled_gt_cohort200_20260716_011804/tiled_gt_manifest.parquet')
MASTER = Path('/nfs-medical3/zyh/fewclick_v2/extensible_prompt/tissue_spatial_split_0922_001235/patches_index.parquet')
CATALOG = Path('/nfs-medical3/zyh/fewclick_v2/paper_multiscale/rerun_1025_033600/training_catalog_1025_040737/tiles.parquet')
PALETTE = Path('benchmarks/v4/phase_1_multiscale/configs/phase1_10x_baseline.yaml')
Image.MAX_IMAGE_PIXELS = None
pyvips.cache_set_max(0)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def canonical(path):
    # Stored JSON sometimes has literal unicode escapes, in addition to JSON escaping.
    if '\\u' in path:
        import re
        path = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), path)
    return str(Path(path).resolve())


def pitch(path):
    with Image.open(path) as image:
        unit = int(image.tag_v2.get(296, 0))
        require(unit in (2, 3), 'physical TIFF resolution unit')
        factor = 25400 if unit == 2 else 10000
        mx, my = [factor/float(image.tag_v2[k]) for k in (282, 283)]
        require(all(math.isfinite(v) and .05<v<5 for v in (mx,my)), 'physical TIFF pitch')
        return image.size, (mx,my)


def union(intervals):
    total = 0; end = None
    for a,b in sorted(set(intervals)):
        total += b-a if end is None or a>end else max(0,b-end)
        end = b if end is None else max(end,b)
    return total


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); worker = os.getpid()
    try:
        inputs = {str(p):sha(p) for p in (COHORT,TILED,MASTER,CATALOG,PALETTE,Path(__file__),Path('benchmarks/v5/wsi_global/plan.py'))}
        cohort = pd.read_parquet(COHORT); tests = cohort[cohort.split=='test'].sort_values('wsi_id').to_dict('records')
        require(len(tests)==30 and len({r['wsi_id'] for r in tests})==30, 'all30 original test slides')
        tiled = pd.read_parquet(TILED).set_index('wsi_id').to_dict('index')
        master = pd.read_parquet(MASTER); catalog = pd.read_parquet(CATALOG)
        ids = set(master.wsi_id.astype(str)); paths = set(map(canonical, master.wsi_path.unique()))
        crc = catalog[catalog.dataset=='crc']
        current_ids = {s.split('|',1)[1] for s in crc.space_key}
        current_paths = {canonical(json.loads(s)['source_image']) for s in crc.multiscale_json}
        require(not ({r['wsi_id'] for r in tests} & (ids | current_ids)), 'test IDs outside current declared input sources')
        require(not ({canonical(r['wsi_path']) for r in tests} & (paths | current_paths)), 'test full4x source native paths outside current sources')
        if args.gate:
            most = max(tests, key=lambda r: max(abs(r['level0_width']-2*r['gt_width']),abs(r['level0_height']-2*r['gt_height'])))
            chosen = {tests[0]['wsi_id'],most['wsi_id']}; tests = [r for r in tests if r['wsi_id'] in chosen]
            require(len(tests)==2, 'exact and discrepant mapping representative gate')
        if not args.gate:
            require(args.gate_evidence is not None, 'real full annotation gate required')
            g = json.loads((args.gate_evidence/'summary.json').read_text())
            require(g['status']=='passed_main30_source_gate' and g['input_sha256']==inputs, 'same-source real gate')
        estimated = sum(4*int(r['gt_width'])*int(r['gt_height']) for r in tests)
        free = shutil.disk_usage(out).free; require(free > 1.5*estimated, 'raw RGB and labels immutable disk headroom')
        write(out/'contract.json',dict(worker_pid=worker,input_sha256=inputs,slides=[r['wsi_id'] for r in tests],gate=args.gate,
            estimated_bytes=estimated,free_bytes=free,GT_mapping='inherited zero-offset extent affine sx=W/GW, sy=H/GH; anatomical registration not independently proven',
            prediction_domain='image geometry only; true4x complete domain; no GT/tissue/grid filtering',failure_policy='stop and preserve, no retry/fallback'))
        classes = yaml.safe_load(PALETTE.read_text())['data']['class_map']
        codes = np.array([(c['rgb'][0]<<16)+(c['rgb'][1]<<8)+c['rgb'][2] for c in classes], dtype=np.uint32)
        order = np.argsort(codes); sorted_codes = codes[order]; sorted_ids = np.array([c['id'] for c in classes],dtype=np.uint8)[order]
        results = []
        for number,row in enumerate(tests):
            for p,value in inputs.items():
                require(sha(p)==value, 'frozen audit input')
            tick = time.perf_counter(); sid=row['wsi_id']; folder=out/sid;folder.mkdir()
            gtpath=Path(row['gt_path']); tilepath=Path(tiled[sid]['tiled_gt_path']); imagepath=Path(row['wsi_path'])
            require(canonical(tiled[sid]['source_gt_path'])==canonical(str(gtpath)), 'original GT lineage')
            size, mpp = pitch(imagepath); w,h=size;gw,gh=int(row['gt_width']),int(row['gt_height'])
            require(size==(int(row['level0_width']),int(row['level0_height'])), 'actual TIFF dimensions')
            native=pyvips.Image.new_from_file(str(imagepath),access='random')
            require((native.width,native.height)==size and native.bands==3, 'native RGB source')
            original=pyvips.Image.new_from_file(str(gtpath),access='sequential')
            cached=pyvips.Image.new_from_file(str(tilepath),access='random')
            require(all((im.width,im.height,im.bands,im.format)==(gw,gh,3,'uchar') for im in (original,cached)), 'actual RGB annotation dimensions')
            sx,sy=w/gw,h/gh
            require(sx==row['inferred_gt_downsample_x'] and sy==row['inferred_gt_downsample_y'], 'inherited ratio exactly retained')
            info=geometry(w,h,*mpp);tiles=list(grid(info));fw,fh=info['fine_native_wh'];pairs=set();xi=[];yi=[]
            with (folder/'grid.jsonl').open('x') as stream:
                for t in tiles:
                    for k,box in zip((1,2,4),t['native_boxes']):
                        x,y,bw,bh=box
                        require(min(x,y)>=0 and x+bw<=w and y+bh<=h and (bw,bh)==(k*fw,k*fh), 'all native input footprints')
                        require(max(abs(x+bw/2-t['centre_level0'][0]),abs(y+bh/2-t['centre_level0'][1]))<=.5, 'concentric views')
                    x,y,bw,bh=t['native_boxes'][0];require((x,y) not in pairs,'unique grid')
                    pairs.add((x,y));xi.append((x,x+bw));yi.append((y,y+bh));stream.write(json.dumps(t)+'\n')
            require(len(pairs)==len(set(xi))*len(set(yi))==info['tiles'], 'complete Cartesian grid')
            require(union(xi)*union(yi)==info['covered_fine_pixels'], 'independent interval union')
            bx0,by0,bx1,by1=info['valid_fine_bounds_xyxy']
            gx0,gy0,gx1,gy1=math.ceil(bx0/sx),math.ceil(by0/sy),math.floor(bx1/sx),math.floor(by1/sy)
            require(gx0*sx>=bx0 and gy0*sy>=by0 and gx1*sx<=bx1 and gy1*sy<=by1, 'full GT pixel area inside predicted domain')
            rawpath=folder/'original_gt_rgb.raw';original.rawsave(str(rawpath))
            require(rawpath.stat().st_size==gw*gh*3,'full raw byte count')
            raw=np.memmap(rawpath,mode='r',dtype=np.uint8,shape=(gh,gw,3))
            labels=np.lib.format.open_memmap(folder/'labels.npy',mode='w+',dtype=np.uint8,shape=(gh,gw))
            counts=np.zeros(256,dtype=np.int64);domain=np.zeros(256,dtype=np.int64);band_count=0
            for y in range(0,gh,256):
                bh=min(256,gh-y);rgb=np.asarray(raw[y:y+bh]);cached_rgb=np.frombuffer(cached.crop(0,y,gw,bh).write_to_memory(),dtype=np.uint8).reshape(bh,gw,3)
                require(np.array_equal(rgb,cached_rgb),'all pixels original PNG versus cached TIFF')
                packed=(rgb[:,:,0].astype(np.uint32)<<16)+(rgb[:,:,1].astype(np.uint32)<<8)+rgb[:,:,2]
                index=np.searchsorted(sorted_codes,packed);safe=np.minimum(index,len(codes)-1)
                valid=(index<len(codes)) & (sorted_codes[safe]==packed)
                decoded=np.where(valid,sorted_ids[safe],255).astype(np.uint8);labels[y:y+bh]=decoded
                counts+=np.bincount(decoded.ravel(),minlength=256)
                lo,hi=max(y,gy0),min(y+bh,gy1)
                if lo<hi:domain+=np.bincount(decoded[lo-y:hi-y,gx0:gx1].ravel(),minlength=256)
                band_count+=1
                if hasattr(raw._mmap,'madvise'):raw._mmap.madvise(mmap.MADV_DONTNEED)
                write(out/'status.json',dict(status='full_GT_pixel_replay',worker_pid=worker,completed_slides=number,total_slides=len(tests),slide=sid,band=band_count,total_bands=math.ceil(gh/256),seconds=time.perf_counter()-started))
            labels.flush();del raw,labels
            historical={int(k):int(v) for k,v in row['class_pixel_counts'].items()}
            require(all(int(counts[c['id']])==historical.get(c['id'],0) for c in classes),'fresh full palette counts versus original cohort')
            require(int(counts[255])==int(row['unknown_pixel_count']) and int(counts[:12].sum())==int(row['valid_pixels']), 'full GT unknown/valid counts')
            require(int(counts.sum())==gw*gh and int(domain.sum())==(gx1-gx0)*(gy1-gy0), 'pixel area conservation')
            probes=[];middle=tiles[len(tiles)//2]
            for scale,box in enumerate(middle['native_boxes']):
                crop=native.crop(*box);image=crop.resize(512/box[2],vscale=512/box[3],kernel='linear');require((image.width,image.height)==(512,512),'real native RGB probe')
                p=folder/f'native_probe_scale{scale}.png';image.pngsave(str(p));probes.append(dict(path=str(p),sha256=sha(p),native_box=box))
            files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (imagepath,gtpath,tilepath,rawpath,folder/'labels.npy',folder/'grid.jsonl')}
            result=dict(wsi_id=sid,patient_id_as_recorded=str(row['patient_id']),split='test',source_wsi=str(imagepath),original_gt=str(gtpath),tiled_gt=str(tilepath),
                native_wh=[w,h],gt_wh=[gw,gh],native_mpp=list(mpp),GT_to_native_affine=[sx,0,0,0,sy,0],
                fixed2_vs_extent_boundary_displacement_native=[w-2*gw,h-2*gh],mapping_qualification='inherited dimension-ratio affine, not independently proven anatomical alignment',
                geometry=info,grid=str(folder/'grid.jsonl'),GT_full_cell_domain_xyxy=[gx0,gy0,gx1,gy1],class_counts={str(c['id']):int(counts[c['id']]) for c in classes},
                domain_class_counts={str(c['id']):int(domain[c['id']]) for c in classes},unknown_pixels=int(counts[255]),domain_unknown_pixels=int(domain[255]),
                pixel_replayed=gw*gh,probes=probes,files=files,seconds=time.perf_counter()-tick)
            results.append(result);write(folder/'audit.json',result)
            write(out/'slides_private.json',results);print(json.dumps(dict(slide=sid,completed=number+1,total=len(tests),pixels=gw*gh,seconds=result['seconds'])),flush=True)
        duplicate=defaultdict(list)
        for r in results:duplicate[r['files'][r['source_wsi']]['sha256']].append(r['wsi_id'])
        summary=dict(status='passed_main30_source_gate' if args.gate else 'complete_main30_full_GT_and_geometry_audit_qualified',worker_pid=worker,gate=args.gate,
            slides=len(results),input_sha256=inputs,slides_sha256=sha(out/'slides_private.json'),total_GT_pixels=sum(r['pixel_replayed'] for r in results),total_tiles=sum(r['geometry']['tiles'] for r in results),
            duplicate_native_file_SHA_groups=[v for v in duplicate.values() if len(v)>1],current_declared_source_ID_intersections=0,current_full4x_native_path_intersections=0,
            class_domain_pixels={str(c['id']):sum(r['domain_class_counts'][str(c['id'])] for r in results) for c in classes},
            total_wall_seconds=time.perf_counter()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            limitations=['No model quality evaluation','Not semantic registration proof; inherited extent affine explicit','Clinical patient map not certified','Current source ID/native paths only, no all-rescan/perceptual duplicate or upstream-pretraining proof','GT is audit/scoring-only; grid image-geometry-only'])
        write(out/'summary.json',summary);write(out/'status.json',dict(status=summary['status'],worker_pid=worker,slides=len(results)))
        print(json.dumps(dict(status=summary['status'],slides=len(results),seconds=summary['total_wall_seconds'])),flush=True)
    except Exception:
        write(out/'status.json',dict(status='failed',worker_pid=worker,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gate',action='store_true');parser.add_argument('--gate-evidence',type=Path)
    run(parser.parse_args())
