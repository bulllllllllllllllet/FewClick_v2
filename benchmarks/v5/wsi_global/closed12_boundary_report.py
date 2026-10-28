"""Descriptive Dice/BF1 for accepted native closed12 WSIs, no paired CIs."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from benchmarks.v5.multidomain_data.prepare import LABELS


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def average(values):
    values=[v for v in values if v is not None]
    return sum(values,Fraction())/len(values) if values else None


def measures(row):
    b=row['physical_BF1'];p=b['predicted_boundary_pixels'];t=b['true_boundary_pixels']
    require(b['boundary_evaluable']==(t>0),'boundary denominator eligibility')
    if not t:f1=None
    elif not p:f1=Fraction()
    else:
        precision=Fraction(b['matched_predicted_boundary_pixels'],p);recall=Fraction(b['matched_true_boundary_pixels'],t)
        f1=2*precision*recall/(precision+recall) if precision+recall else Fraction()
    actual=b['bf1_um2']
    require((f1 is None)==(actual is None) and (f1 is None or abs(float(f1)-actual)<2e-15),'each independently accepted BF1 scalar from exact counts')
    dice=Fraction(2*row['tp'],2*row['tp']+row['fp']+row['fn']) if row['positive'] else None
    require((dice is None)==(row['dice'] is None) and (dice is None or float(dice)==row['dice']),'exact accepted integer Dice counts')
    return dict(dice=dice,bf1_um2=f1)


def describe(rows):
    by_wsi=defaultdict(list)
    for row in rows:by_wsi[row['wsi_id']].append(measures(row))
    per_wsi=[dict(wsi_id=sid,**{k:average(v[k] for v in values) for k in ('dice','bf1_um2')}) for sid,values in sorted(by_wsi.items())]
    return dict(pairs=len(rows),WSIs=len(by_wsi),boundary_evaluable=sum(r['physical_BF1']['boundary_evaluable'] for r in rows),GT_boundary_NA=sum(not r['physical_BF1']['boundary_evaluable'] for r in rows),
        predicted_boundary_empty_with_GT=sum(r['physical_BF1']['boundary_evaluable'] and not r['physical_BF1']['predicted_boundary_pixels'] for r in rows),
        WSI_equal={k:average(v[k] for v in per_wsi) for k in ('dice','bf1_um2')},pair_equal={k:average(measures(r)[k] for r in rows) for k in ('dice','bf1_um2')},per_WSI=per_wsi)


def jsonable(value):
    if isinstance(value,Fraction):return float(value)
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,list):return [jsonable(v) for v in value]
    return value


def run(a):
    root=a.root;boundary=root/'WSI_boundary_closed12_complete_1028_040500';source=root/'closed12_raster_full_1027_221200'
    bs=read(boundary/'summary.json');native=read(source/'summary.json');receipt=read(root/'closed12_raster_acceptance_1027_221200/summary.json')
    require(bs['status']=='accepted_complete_WSI_physical_BF1_rows' and bs['rows']==360 and bs['signature']['kind']=='closed12','actual complete360 independent BF1 join')
    require(bs['prediction_summary_sha256']==receipt['source_summary_sha256']==sha(source/'summary.json') and receipt['status']=='accepted_closed12_full30_native_raster_and_GT_scores','same independently accepted native30 outputs')
    require(sha(boundary/'rows.json')==bs['rows_sha256'] and sha(source/'rows.json')==native['rows_sha256'],'all immutable joined/native rows')
    for path in [boundary,root/'boundary_accept_queue_1028_040500']:
        require(Path(str(path)+'.log').read_text().splitlines()[-1]=='EXIT_CODE=0','actual terminal aggregate/audit coordinator receipts')
    for evidence in bs['evidence']:
        for name,key in [('source','source_summary_sha256'),('acceptance','acceptance_summary_sha256')]:require(sha(Path(evidence[name])/'summary.json')==evidence[key],'all four independent shard receipts')
        require(read(Path(evidence['acceptance'])/'summary.json')['status']=='accepted_WSI_boundary_shard','all four original-GT disk audits passed')
    rows=read(boundary/'rows.json');original=read(source/'rows.json')
    require(len(rows)==360 and len({(r['wsi_id'],r['target']) for r in rows})==360,'all30 by12 unique rows')
    for row,old in zip(rows,original):require({k:row[k] for k in old}==old,'native class/GT counts and all fields unchanged by join')
    protocol=root/'support_query_full_1027_173800';ps=read(protocol/'summary.json')
    require(sha(protocol/'summary.json')==native['signature']['protocol_sha256'] and sha(protocol/'coverage.json')==ps['output_sha256']['coverage.json'],'fixed pre-prediction support availability')
    available={(r['wsi_id'],r['target']) for r in read(protocol/'coverage.json') if r['status']=='available'}
    tissue=[r for r in rows if r['target']!=2];matched=[r for r in tissue if (r['wsi_id'],r['target']) in available]
    require((len(tissue),len(matched))==(330,276),'all tissues and fixed comparison subset retained')
    categories=[dict(target=c,label=LABELS['crc'][c],values=describe([r for r in rows if r['target']==c])) for c in range(12)]
    result=jsonable(dict(status='descriptive_accepted_closed12_Dice_BF1_report',worker_pid=os.getpid(),source_sha256=sha(__file__),native_acceptance_sha256=sha(root/'closed12_raster_acceptance_1027_221200/summary.json'),boundary_summary_sha256=sha(boundary/'summary.json'),all330=describe(tissue),supported276=describe(matched),background30=describe([r for r in rows if r['target']==2]),per_class=categories,
        aggregation='Exact count-derived rational scalar/means; classes averaged within WSI then WSIs equally; pair-equal mean separate. BF1 undefined GT boundary stays NA; predicted boundary empty with GT stays0.',
        scope='CurrentCRC30 strict-remote closed12 full-class CE/target-ID reference without clicks; independent physical Euclidean2um BF1. No paired CI, patient/seed independence or FewClick comparison claim.'))
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 闭集全监督参考：已接受的Dice与物理BF1','',
        '固定30WSI严格远距域；12类CE/组织ID/无点击的闭集参考，与交互方法分表。BF1@2µm使用原GT物理像元间距和独立欧氏圆盘复算，背景ID2另表。','',
        '|范围|组织—WSI对|可评BF1|GT边界NA|WSI等权Dice|WSI等权BF1@2µm|','|---|---:|---:|---:|---:|---:|']
    for name,key in [('全部11组织','all330'),('固定支持可用子集','supported276')]:
        v=result[key];lines.append(f'|{name}|{v["pairs"]}|{v["boundary_evaluable"]}|{v["GT_boundary_NA"]}|{v["WSI_equal"]["dice"]:.6f}|{v["WSI_equal"]["bf1_um2"]:.6f}|')
    lines+=['','无GT边界保留NA；有GT边界但预测无边界计0。所有标签/切片及54不可用支持组合保留；276子集按预测前固定可用性形成。先每WSI平均组织再对WSI等权，与直接平均组织—WSI对不同；此处无配对CI或方法胜负结论。','',
        '|ID|标签|30片均值Dice|30片均值BF1@2µm|预测边界空且GT边界存在|','|---|---|---:|---:|---:|']
    for c in result['per_class']:
        v=c['values'];lines.append(f'|{c["target"]}|{c["label"]}|{v["pair_equal"]["dice"]:.6f}|{v["pair_equal"]["bf1_um2"]:.6f}|{v["predicted_boundary_empty_with_GT"]}|')
    lines+=['',f'完整BF1接受：{boundary}/summary.json，SHA {result["boundary_summary_sha256"]}。360行均已独立原GT/预测计数/块守恒复核；本报告由原整数计数exact rational重构标量与派生均值，不重新推理或改变指标。','',
        '原空间block与extent affine资格保留；主30交互方法/完整细胞链、组件/距离与配对统计和完整有效域诊断仍分别等待，不将当前闭集结果归于FewClick。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ('status','worker_pid','source_sha256','boundary_summary_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
