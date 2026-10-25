"""Recount the complete tile Cartesian product and rectangle-union coverage."""
import argparse
import json
from pathlib import Path

from benchmarks.v5.fair_comparison.audit import digest,write


def interval_union(intervals):
    end=None;total=0
    for left,right in sorted(intervals):
        if end is None or left>end:total+=right-left
        elif right>end:total+=right-end
        end=right if end is None else max(end,right)
    return total


def run(a):
    a.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads((a.plan/'summary.json').read_text());rows=json.loads((a.plan/'slides_private.json').read_text())
    if summary['status']!='complete_spatial_plan' or digest(a.plan/'slides_private.json')!=summary['slides_sha256']:
        raise ValueError('spatial plan incomplete or changed')
    if summary['slides']!=30 or len(rows)!=30:raise ValueError('all30 external TLS slides required')
    results=[]
    for row in rows:
        path=Path(row['grid']);info=row['geometry'];w,h=row['width'],row['height'];fw,fh=info['fine_native_wh']
        if digest(path)!=row['grid_sha256']:raise ValueError('grid SHA changed')
        tiles=[json.loads(s) for s in path.open()];xs=set();ys=set();pairs=set();xi=[];yi=[]
        for number,tile in enumerate(tiles):
            if tile['number']!=number:raise ValueError('tile numbering')
            (x,y,bw,bh),middle,coarse=tile['native_boxes']
            if (bw,bh)!=(fw,fh):raise ValueError('fine dimensions')
            for factor,(bx,by,bw,bh) in zip((1,2,4),tile['native_boxes']):
                if (bw,bh)!=(factor*fw,factor*fh) or min(bx,by)<0 or bx+bw>w or by+bh>h:raise ValueError('context footprint')
                if max(abs(bx+bw/2-tile['centre_level0'][0]),abs(by+bh/2-tile['centre_level0'][1]))>.500001:
                    raise ValueError('concentric integer context')
            if (x,y) in pairs:raise ValueError('duplicate tile')
            pairs.add((x,y));xs.add(x);ys.add(y);xi.append((x,x+fw));yi.append((y,y+fh))
        if len(pairs)!=len(xs)*len(ys) or len(pairs)!=info['tiles']:raise ValueError('incomplete Cartesian grid')
        union_x,union_y=interval_union(xi),interval_union(yi)
        if union_x!=max(xs)+fw-min(xs) or union_y!=max(ys)+fh-min(ys):raise ValueError('fine coverage gaps')
        area=union_x*union_y
        if area!=info['covered_fine_pixels'] or w*h-area!=info['uncovered_edge_pixels']:raise ValueError('coverage area accounting')
        results.append(dict(slide_id=row['slide_id'],tiles=len(tiles),covered_fine_pixels=area,
            uncovered_edge_pixels=w*h-area,covered_area_fraction=area/(w*h)))
    if sum(r['tiles'] for r in results)!=summary['total_tiles']:raise ValueError('cohort tile count')
    write(a.output/'summary.json',dict(status='passed_full_spatial_plan_audit',slides=len(results),total_tiles=summary['total_tiles'],
        slides_sha256=summary['slides_sha256'],source_sha256=digest(__file__),results=results,
        scope='All30 grid file SHA, every input footprint/centre, complete Cartesian product and rectangle union; no model accuracy'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
