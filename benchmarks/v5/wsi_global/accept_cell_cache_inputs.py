"""Independently bind the full XCell input manifest to all frozen native views."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import time

from PIL import Image


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def require(test,why):
    if not test:
        raise ValueError(why)


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();pid=os.getpid()
    source=read(a.source/'summary.json');require(source['status']=='complete_main30_fullcell_input_manifest','complete all30 input preparation')
    protocol=a.root/'support_query_full_1027_173800'
    require(source['signature']['source_sha256']==sha('benchmarks/v5/wsi_global/cell_cache_inputs.py') and source['signature']['protocol_sha256']==sha(protocol/'summary.json'),'frozen input recipe and original protocol')
    require(sha(a.source/'inputs.jsonl')==source['input_manifest_sha256'],'entire native input manifest bytes')
    entries=[json.loads(line) for line in (a.source/'inputs.jsonl').open()]
    require(len(entries)==20456 and [e['index'] for e in entries]==list(range(20456)),'all ordered canonical input slots')
    by_key={e['key']:e for e in entries};require(len(by_key)==20456,'all source/query input keys unique')
    expected=[];views=0;source_files=0;odd_slide_ids=set();evidence=[]
    for row in read(protocol/'query_inputs.json'):
        sid=row['wsi_id'];w,h=row['native_wh'];require(sha(row['grid'])==row['grid_sha256'],'all native remote grid bytes')
        sources=[];queries=[]
        for support in row['support_inputs']:
            key=support['support_id'];boxes=[d['native_box'] for d in support['images']];sources.append((key,boxes,support['images']))
        for tile in (json.loads(line) for line in Path(row['grid']).open()):
            queries.append((sid+':query:'+str(tile['number']),tile['native_boxes'],tile['number']))
        for kind,items in [('support',sources),('query',queries)]:
            for key,boxes,metadata in items:
                expected.append(key);e=by_key[key]
                require((e['wsi_id'],e['kind'],e['source_wsi'],e['native_wh'],e['native_mpp'],e['native_boxes'])==(sid,kind,row['source_wsi'],row['native_wh'],row['native_mpp'],boxes),'all exact original native view/slide/physical pitch identities')
                require(e['source_cut_native']==row['source_cut_native'] and e['gap_native']==row['gap_native'],'unchanged physical source/query policy')
                require(len(boxes)==3 and all(0<=x and 0<=y and x+bw<=w and y+bh<=h for x,y,bw,bh in boxes),'all three full input footprints in native slide')
                fine=boxes[0];centre=(fine[0]+fine[2]/2,fine[1]+fine[3]/2)
                require(all(max(abs(x+bw/2-centre[0]),abs(y+bh/2-centre[1]))<=.5 for x,y,bw,bh in boxes),'original integer-rounding view-centre policy')
                if fine[2]%2 or fine[3]%2:odd_slide_ids.add(sid)
                if kind=='support':
                    require(boxes[-1][0]+boxes[-1][2]<=row['source_cut_native'] and e['HE_reference']==metadata,'all support contexts/references unchanged')
                    for d in metadata:
                        require(sha(d['path'])==d['sha256'],'every sourceHE512 PNG byte')
                        with Image.open(d['path']) as im:require(im.size==(512,512) and im.mode=='RGB','all actual source HE RGB shapes')
                        source_files+=1
                else:
                    require(boxes[-1][0]>=row['source_cut_native']+row['gap_native'] and e['tile']==metadata and e['HE_reference'] is None,'all full4x remote queries unchanged, no GT reference')
                views+=3
        evidence.append(dict(wsi_id=sid,support_inputs=len(sources),query_inputs=len(queries)))
        (a.output/'status.json').write_text(json.dumps(dict(status='independent_fullcell_native_manifest_geometry',worker_pid=pid,slides_complete=len(evidence),total=30))+'\n')
    require([e['key'] for e in entries]==expected and views==61368 and source_files==744 and len(evidence)==30,'all full cohort ordering/views/support RGB files accounted')
    require(source['counts']==dict(support=248,query=20208),'complete fullcell input counts')
    result=dict(status='accepted_full30_fullcell_native_input_manifest',worker_pid=pid,inputs=len(entries),views=views,source_HE_files=source_files,odd_native_width_or_height_slides=sorted(odd_slide_ids),evidence=evidence,source_summary_sha256=sha(a.source/'summary.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-start,
                scope='Every fixed support/query record, native boxes/MPP/order/full4x separation and all744 support HE RGB files; no query GT. Fresh native one-read equality is covered by the separate24-view gate, not exhaustively repeated here',remaining='Actual InstanSeg/CTP/XCell preparation and encoder/cache/decode/quality gates; no nuclei/features/results claimed')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','root','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
