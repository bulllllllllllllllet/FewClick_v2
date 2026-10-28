"""Descriptive closed12 WSI report, gated on actual full native acceptance.

No inference, image/label array access, CI or comparative superiority claim.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from benchmarks.v5.multidomain_data.prepare import LABELS


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def mean(values):
    values = [float(v) for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None


def describe(rows):
    by_slide = defaultdict(list)
    for r in rows:
        by_slide[r['wsi_id']].append(r)
    per_slide = [dict(wsi_id=sid, pairs=len(group), positive_pairs=sum(r['positive']>0 for r in group),
                      present_tissue_macro_dice=mean(r['dice'] for r in group),
                      absent_target_mean_false_positive_fraction=mean(r['false_positive_fraction'] for r in group if not r['positive']))
                 for sid, group in sorted(by_slide.items())]
    return dict(pairs=len(rows), slides=len(by_slide), WSIs_with_positive_pairs=sum(r['positive_pairs']>0 for r in per_slide),
                WSIs_without_positive_pairs=sum(r['positive_pairs']==0 for r in per_slide), positive_pairs=sum(r['positive']>0 for r in rows),
                absent_pairs=sum(r['positive']==0 for r in rows),
                WSI_equal_present_tissue_macro_dice=mean(r['present_tissue_macro_dice'] for r in per_slide),
                pair_equal_present_tissue_macro_dice=mean(r['dice'] for r in rows),
                absent_target_mean_false_positive_fraction=mean(r['false_positive_fraction'] for r in rows if not r['positive']),
                per_WSI=per_slide)


def run(a):
    root = a.root; source = root/'closed12_raster_full_1027_221200'; accepted = root/'closed12_raster_acceptance_1027_221200'
    s = read(source/'summary.json'); receipt = read(accepted/'summary.json')
    require(receipt['status']=='accepted_closed12_full30_native_raster_and_GT_scores' and receipt['items']==30 and receipt['source_summary_sha256']==sha(source/'summary.json'), 'actual full30 independent native/GT acceptance')
    for folder in (source, accepted):
        require('\nEXIT_CODE=0\n' in Path(str(folder)+'.log').read_text(), 'successful actual terminal process receipt')
    require(s['status']=='complete_closed12_WSI_quality_execution_unaccepted' and s['rows']==360 and s['items']==30 and s['tiles']==20208, 'all fixed native query fields and class rows executed')
    require(sha(source/'rows.json')==s['rows_sha256'] and sha(source/'items.json')==s['items_sha256'], 'immutable complete output metadata')
    for p,h in s['signature']['source_sha256'].items():
        require(sha(p)==h, 'accepted native producer/kernel unchanged')
    protocol = root/'support_query_full_1027_173800'; ps = read(protocol/'summary.json')
    require(sha(protocol/'summary.json')==s['signature']['protocol_sha256'], 'same original support/remote protocol')
    for name,h in ps['output_sha256'].items():
        require(sha(protocol/name)==h, 'original availability and support selection frozen')
    rows = read(source/'rows.json'); items = read(source/'items.json')
    require(len(rows)==len({(r['wsi_id'],r['target']) for r in rows})==360 and len({r['wsi_id'] for r in rows})==30, 'all30 by12 classes once')
    coverage = read(protocol/'coverage.json'); available = {(c['wsi_id'],c['target']) for c in coverage if c['status']=='available'}
    require(len(available)==276 and len(coverage)==330, 'unchanged276 supported and54 unavailable pairs')
    names = LABELS['crc']; require(names[2]=='background' and names[10]=='fat' and set(names)==set(range(12)), 'original CRC label mapping')
    # Independently derive report counts from the already independently accepted
    # confusion matrices. No stored scalar or row order defines its denominator.
    matrices = {i['wsi_id']:i['confusion'] for i in items}
    for row in rows:
        m = matrices[row['wsi_id']]; c = row['target']; tp = int(m[c][c]); truth = sum(m[c]); predicted = sum(line[c] for line in m)
        valid = sum(map(sum,m)); fp = predicted-tp; fn = truth-tp
        require((tp,fp,fn,truth,valid)==tuple(row[k] for k in ('tp','fp','fn','positive','valid')), 'every accepted integer denominator recounted')
        require(row['dice']==(2*tp/(truth+predicted) if truth else None) and row['false_positive_fraction']==fp/valid, 'exact Dice/absent false-positive eligibility')
    tissue = [r for r in rows if r['target']!=2]; matched = [r for r in tissue if (r['wsi_id'],r['target']) in available]
    require(len(tissue)==330 and len(matched)==276 and {r['target'] for r in tissue}==set(names)-{2}, 'all tissues including fat; no ranking/sample filtering')
    categories = []
    for target in range(12):
        all_class = [r for r in rows if r['target']==target]; selected = [r for r in all_class if (r['wsi_id'],r['target']) in available]
        categories.append(dict(target=target,label=names[target],all30=describe(all_class),supported_subset=describe(selected) if target!=2 else None))
    result = dict(status='descriptive_accepted_closed12_native_WSI_report',worker_pid=os.getpid(),source_sha256=sha(__file__),
        source_summary_sha256=sha(source/'summary.json'),acceptance_summary_sha256=sha(accepted/'summary.json'),label_mapping_sha256=sha('benchmarks/v5/multidomain_data/prepare.py'),
        weight_sha256=read(root/'closed12_full_cache_1027_212000/contract.json')['signature']['weight_sha256'],
        scope='Current CRC30 fixed strict-remote native output, closed12 full-class CE and target ID privilege, no clicks; not interactive/few-click quality or equal-information control.',
        all330_tissue=describe(tissue),supported276_tissue=describe(matched),background30=describe([r for r in rows if r['target']==2]),per_class=categories,
        native_pixels=receipt['pixels'],valid_GT_pixels=sum(sum(map(sum,m)) for m in matrices.values()),
        aggregation='Present-tissue Dice: average available class Dice within each WSI, then WSIs equally; additionally pair-equal mean. Absent target Dice stays NA with false positives reported; all pairs retained. No CIs until paired WSI statistics.',
        remaining='BF1/component/distance acceptance, complete interactive and fullcell main30 scores, paired WSI uncertainty and complete-valid4x diagnostic.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 当前CRC30闭集监督参考：原生远距输出', '',
           '本表使用12类CE监督及组织ID，不使用点击；作为特权闭集参考，与交互模型分表。30张原固定切片、20208远距视野，无删样本或类别，脂肪正常训练。', '',
           '|范围|组织—WSI对|GT有目标|GT无目标|WSI等权平均Dice|对等权平均Dice|',
           '|---|---:|---:|---:|---:|---:|']
    for title,key in [('全部组织','all330_tissue'),('固定支持可用子集','supported276_tissue')]:
        d=result[key];lines.append(f'|{title}|{d["pairs"]}|{d["positive_pairs"]}|{d["absent_pairs"]}|{d["WSI_equal_present_tissue_macro_dice"]:.6f}|{d["pair_equal_present_tissue_macro_dice"]:.6f}|')
    lines += ['', 'Dice仅在GT有目标时可评；GT无目标保留为NA并单列误报，不能把NA填0或1。先每WSI平均组织再平均WSI，与把所有组织—WSI对直接平均的权重不同。若WSI全部目标NA，其均值也为NA，可评/不可评WSI数量完整保存在summary.json。该表为描述性摘要，尚无配对CI或模型胜负结论。', '',
              '|ID|标签|30张中GT有目标|均值Dice（GT有目标）|GT无目标数|无目标平均FP/valid|', '|---|---|---:|---:|---:|---:|']
    for c in categories:
        d=c['all30'];dice=d['pair_equal_present_tissue_macro_dice'];fp=d['absent_target_mean_false_positive_fraction']
        lines.append(f'|{c["target"]}|{c["label"]}|{d["positive_pairs"]}|{dice if dice is not None else "NA"}|{d["absent_pairs"]}|{fp if fp is not None else "NA"}|')
    lines += ['',f'接受凭据：{accepted}/summary.json，SHA {result["acceptance_summary_sha256"]}。原始rows与矩阵SHA已核对；主表排除背景ID2，背景结果完整保存在summary.json。', '',
              '原GT extent affine与空间block来源政策保留；没有独立解剖配准、病例独立或上游预训练全暴露证明。当前矩形域不代表100%组织覆盖，完整有效域仍单独评价。', '',
              '待完成：完整方法输出/BF1/组件距离与配对WSI统计；本报告不替代原论文checkpoint复现或所有审稿问题闭环。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_class','all330_tissue','supported276_tissue','background30')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
