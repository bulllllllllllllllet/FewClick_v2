"""Plan every full-context-valid tile without query labels or tissue filtering."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd

from benchmarks.v5.external_evaluation.wsi_reader import StrictSlide
from benchmarks.v5.fair_comparison.audit import digest, write


def axis_origins(length, fine):
    margin = math.ceil(1.5*fine)
    start, last = margin, length-margin-fine
    if last < start:raise ValueError('slide too small for three physical views')
    values = list(range(start,last+1,fine))
    if values[-1] != last:values.append(last)
    return values


def geometry(width, height, mx, my):
    fw, fh = round(256/mx), round(256/my)
    xs, ys = axis_origins(width,fw), axis_origins(height,fh)
    bounds = [xs[0],ys[0],xs[-1]+fw,ys[-1]+fh]
    covered = (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
    return dict(fine_native_wh=[fw,fh],native_mpp=[mx,my],fine_field_um=[fw*mx,fh*my],
        x_origins=xs,y_origins=ys,tiles=len(xs)*len(ys),valid_fine_bounds_xyxy=bounds,
        full_slide_pixels=width*height,covered_fine_pixels=covered,uncovered_edge_pixels=width*height-covered,
        covered_area_fraction=covered/(width*height),stride_native=[fw,fh],
        context_edge_policy='Exclude fine pixels requiring unavailable full4x context; no padding or fabricated views',
        last_row_column='Include last valid origin; final row/column may overlap',tissue_filter=False)


def grid(info):
    fw,fh = info['fine_native_wh'];number = 0
    for y in info['y_origins']:
        for x in info['x_origins']:
            cx,cy = x+fw/2,y+fh/2
            boxes = [[round(cx-fw*k/2),round(cy-fh*k/2),fw*k,fh*k] for k in (1,2,4)]
            yield dict(number=number,centre_level0=[cx,cy],native_boxes=boxes)
            number += 1


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    rows = pd.read_parquet(a.manifest).to_dict('records');records = []
    for index,row in enumerate(rows):
        # Annotation content is intentionally never opened by the planner.
        slide = StrictSlide(row['wsi_path'])
        try:
            info = geometry(slide.width,slide.height,slide.mx,slide.my)
            identifier = 'tls-'+hashlib.sha256(str(row['wsi_path']).encode()).hexdigest()[:16]
            path = a.output/(identifier+'_grid.jsonl')
            with path.open('x') as stream:
                for tile in grid(info):
                    for x,y,w,h in tile['native_boxes']:
                        if min(x,y)<0 or x+w>slide.width or y+h>slide.height:raise ValueError('planned context outside slide')
                    stream.write(json.dumps(tile,sort_keys=True)+'\n')
            for values,extent in [(info['x_origins'],info['fine_native_wh'][0]),(info['y_origins'],info['fine_native_wh'][1])]:
                if any(b>a+extent for a,b in zip(values,values[1:])):raise ValueError('uncovered gap in valid grid')
            records.append(dict(manifest_row=index,slide_id=identifier,width=slide.width,height=slide.height,
                pitch_source=slide.pitch_source,geometry=info,grid=str(path),grid_sha256=digest(path),
                provenance=row))
            write(a.output/'status.json',dict(status='running',worker_pid=os.getpid(),slides=len(records),total_slides=len(rows)))
            print(json.dumps(dict(slide=identifier,tiles=info['tiles'],covered_area_fraction=info['covered_area_fraction'])),flush=True)
        finally:slide.close()
    write(a.output/'slides_private.json',records)
    write(a.output/'summary.json',dict(status='complete_spatial_plan',worker_pid=os.getpid(),slides=len(records),
        total_tiles=sum(r['geometry']['tiles'] for r in records),manifest_sha256=digest(a.manifest),
        slides_sha256=digest(a.output/'slides_private.json'),source_sha256=digest(__file__),
        cohort='External TLS manifest, distinct from main30 test cohort',query_GT_access=False,
        scope='Actual headers and all full-context-valid tiles; no model inference, cells or complete-annotation accuracy'))
    write(a.output/'status.json',dict(status='complete_spatial_plan',worker_pid=os.getpid(),slides=len(records)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
