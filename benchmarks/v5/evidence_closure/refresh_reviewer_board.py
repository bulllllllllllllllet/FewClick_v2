"""Bind all reviewer challenges to accepted new evidence, never to invented edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


# Interpretation is explicit; completed evidence below is checked separately.
UPDATES={
 'ZVeE-1':('L S','新协议适配基线/闭集参考与三seed已独立接受；新local中适配SAM较高。','相同新增标签池/40k任务的对照可以回应适配混杂；非原140WSI/F0复现，不支持当前架构全面优于adapted SAM。','旧投稿模型/原140训练清单与实际上传身份；新主30整片完整结果。'),
 'ZVeE-2':('H P','N/A/L标签留出全训练/统计已接受qualified，原脂肪低分保留；TLS语义/负例覆盖未闭合。','限定为已知训练标签条件检索及已列明标签留出压力测试；脂肪在主12标签中正常监督。','TLS标注完整性/非TLS淋巴聚集难阴性与基线；不能称任意新组织或纯监督删除因果对照。'),
 'ZVeE-3':('I RUN P','主30原GT/全网格/来源和最大4x footprint已接受；完整质量仍在执行/复核。','新cohort预先全30保留，说明strictremote20208与fullvalid28452/支持带；不以patch迁移代替整片。','原5/30选择原因和原清单、完整main30质量/CI及论文绑定。'),
 'ZVeE-4':('X I P','用户已确认XCell在内部数据训练且不含当前CRC/TIGER；原200WSI公开描述仍需补。','披露H&E推理无需mIF而预训练有额外先验；不把来源未知继续当结论，不编造正式引用。','XCell正式出处/自研状态及内部训练资料；原200队列机构/癌种/MPP/清单与上传版本。'),
 'wHd8-1':('C P','当前新模型offline编码/cache、source task、query图解码的操作性定义可追溯；正文未改。','按实际运行版本组织Methods，三视野与Dense另标；192节点是每query窗口，不是全WSI跨tile图。','精确提交版本及图/公式一致改写；不能将实际forward变化说成画图笔误。'),
 'wHd8-2':('L S R','新增监督匹配、public/adapted/闭集和local/transfer分轨已接受。','GT定位信息与训练适配是独立因素，前者不能抵消后者；清洁R模型没有R监督更新，保留该限制。','写清先验/视野/容量/可训练层/选模曝光；不称等总监督或已隔离架构因果效应。'),
 'wHd8-3':('W C','首含细胞3050窗口/9150视野/58真实空视野、两预算全原生概率与成本已qualified接受。','报告冷准备与缓存后完整输出、GPU峰值、实例/节点发生次数；task毫秒不能当整片wall。','当前为固定CRC source→TLS机械样例/shared node，不是TLS Dice/30片平均/受控硬件加速比。'),
 'wHd8-4':('P','本地TeX与PDF可定位.7522全matched集和.7915共同子集解释；local字节已固定。','明确集合/分母和global或pixel-pooled定义；不能把新成绩替换进旧表。','原episode两manifest/885 occurrences819unique及原聚合重算、实际上传版仍缺。'),
 'wHd8-5':('X M G S','none/形态/XCell及普通HE两投影控制、配对统计和三seed已接受；稳定分子因果优势未建立。','Dense对区域头在该固定协议有收益；XCell相对NoCell和whitened HE的区间含0，不能归因mIF目标。','同网络/同数据预训练目标控制若保留独立分子因果主张仍需补；不得删不利控制。'),
 'HVPZ-1':('E','IGNITE真实10中心/9类六方法完整比较、cell输入及原生gate已独立接受。','可报小规模外部域比较并披露各自先验/视野与10中心不确定性；当前FewClick较低。','更多真实物理FOV外部域/中心证据；PUMA/OCELOT小ROI不能伪造1024um上下文。'),
 'HVPZ-2':('R','MultiverSeg稀疏/完整source-mask双轨及837共同子集已完整接受。','披露额外支持mask标注成本，与点击稀疏条件分表；不能将该差异解读为同支持信息。','严格最近365天的方法覆盖仍需文献核验/实验；AI给出的引用不自动视为真实或已比较。'),
 'AI-1':('H I','已知类主12标签与N/A/L留出证据分开。','主张限定条件检索和明确测试标签，保留留出失败。','任意新组织/open-world主张不成立；TLS语义/形态排除仍缺。'),
 'AI-2':('L S R','适配公平性和现有上下文方法对比完成。','按实际训练/支持信息说明系统区别。','SimpleClick/SegNext等AI推荐未在本轮核验，不宣称已纳入或覆盖最近一年。'),
 'AI-3':('I RUN','30片全图8连通组件/5520域已独立接受，组件/距离评分与BF1/统计链已门控并在等待正式结果。','排除命中正提示的目标组件，保留非目标FP，报告0.5/0.1检出及物理距离层；不能报未完成成绩。','完整main30模型预测/组件/距离/BF1接受；定性抽样不可GT选最好点组伪装常规。'),
 'AI-4':('S I P','三seed独立均值/ddof1方差已接受；新WSI以WSI为统计单位，不把5组提示或4预算当独立样本。','报告训练seed与当前来源/WSI条件bootstrap的不同范围。','原5/30选择和提交绑定；完整main30统计未产出，不声称患者独立。'),
 'AI-5':('C','当前top4物理overlap/1e-8有效边、KNN6、双向边/祖先及detach代码可定位。','拓扑稀疏不等于计算稀疏：当前dense masked MultiheadAttention；192节点仅每query三视野。','原图/公式按真实版本改写，不能声称全WSI图或已实现O(E)稀疏kernel。'),
 'AI-6':('L S RUN','新local Dice/BF1和三seed已接受，FewClick仍低于adapted SAM；main30质量待接受。','明确边界/绝对精度局限，避免可靠临床精细分割或优于全部基线断言。','完整WSI误差/边界/距离结果；不以小gate性能当正式成绩。'),
 'AI-7':('C P','实际系统可描述为区域/细胞/多尺度/带符号提示的整合。','只按已实现功能定位贡献，不把成熟模块逐项宣称首创。','相关工作/AI建议引文需逐篇核验，原上传版方法身份未绑定。'),
 'AI-8':('M G X','普通HE PCA64/white64、形态与none/XCell完整控制和准备均接受。','核类别unknown；形态/计数与外观先验分账；不把64维或不同预训练容量当独立分子因果效应。','同网络同数据预训练目标控制仍缺，G6整体partial。'),
 'AI-9':('C P','代码先sigmoid区域概率软投影再pixel阈值；Dense分支在新版本中单列。','图/公式需与实际forward一致。','原Figure1与上传版本需核对，未实际更改论文/图。'),
 'AI-10':('C P','理想sum_k A=1，border双线性采样保持该和；再归一化为数值防护，不是额外生物学模块。','可将数学冗余与实际clamp/浮点条件区分说明。','未改Eq10，不能编造插值必然破坏partition-of-unity的理由。'),
 'AI-11':('C P','log1p estimated region_cell_count，无物理区域面积分母；本地Methods已有count说明。','统一称细胞计数/丰度，不能称面积归一化density。','本地abstract/结论仍用density术语；实际上传版本与修改待绑定。'),
 'AI-12':('L S P','新local pair-macro/class-macro/pixel counts明确；历史全matched/common subset在本地可定位。','列明原分子分母/样本集合，避免macro/micro混名。','原两manifest和全部旧分数重算/上传版未闭合。'),
 'AI-13':('C','当前PromptSetPool显式拒绝空valid set，四个预算均至少1负点。','当前实验应限定至少1有效负点；不要声称已支持或测试0负点。','若保留optional-negative API需另定义/实现/真实门控；本轮未改模型。'),
 'AI-14':('I C','新BF1固定双侧4邻域/真实各轴MPP/Euclidean2um，GT无边界NA、预测无边界但GT有计0。','旧2pixel/方形容差与新物理指标分表，不能当同一口径。','旧物理映射/原表重算及论文指标定义绑定。'),
}


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);root=Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison')
    original=read(a.reviews);require(sha(original['source'])==original['source_sha256'],'actual user saved HTML unchanged')
    for n in original['notes']:require(hashlib.sha256(Path(n['saved_text']).read_text().removesuffix('\n').encode()).hexdigest()==n['text_sha256'],'verbatim original review SHA')
    old=read(a.reviews.parent/'challenge_board.json');fingerprints=read(a.manuscripts)
    for p in fingerprints['files']:require(sha(p['path'])==p['sha256'] and not p['submitted_version_proven'],'all local PDFs/TeX unchanged and no upload identity inferred')
    files={
      'L':['protocol_1025_124208/G3_complete_audit_1025_155140.json'],
      'S':['replication_1025_153837/complete_report_1025_230100/summary.json','replication_1025_153837/independent_aggregate_recount_1025_231100.json'],
      'M':['mechanisms_1025_134753/acceptance_1025_225200/summary_user_provenance.json','mechanisms_1025_134753/acceptance_1025_225200/acceptance.json'],
      'G':['generic_cells_1025_231000/paired_report_1027_141000/summary.json','generic_cells_1025_231000/paired_acceptance_1027_141300/summary.json'],
      'E':['external_1025_143257/complete_report_1025_221700/summary.json','external_1025_143257/audit_1025_221700.json'],
      'R':['transfer_1025_131856/acceptance_1025_224800/summary_user_provenance.json','transfer_1025_131856/acceptance_1025_224800/acceptance.json'],
      'H':['heldout_1026_003000/heldout_statistics_1027_142900/summary.json','heldout_1026_003000/heldout_statistics_acceptance_1027_143400/summary.json'],
      'X':['mechanisms_1025_134753/XCell用户来源确认_1025_140237.json'],
      'W':['wsi_global_1026_010800/fullcell_efficiency_accepted_1028_143000/summary.json','wsi_global_1026_010800/fullcell_output_acceptance_1027_223500/summary.json','wsi_global_1026_010800/fullcell_inputs_acceptance_1027_222500/summary.json'],
      'I':['wsi_global_1026_010800/main30_acceptance_1027_171918/summary.json','wsi_global_1026_010800/support_query_acceptance_1027_173800/summary.json','wsi_global_1026_010800/gt_components_acceptance_1027_231500/summary.json','wsi_global_1026_010800/full_valid_inputs_acceptance_1028_034700/summary.json']}
    evidence={k:[dict(path=str(root/p),sha256=sha(root/p),status=read(root/p).get('status')) for p in paths] for k,paths in files.items()}
    for group,key in [('S','summary_sha256'),('G','report_sha256'),('E','report_sha256'),('H','report_sha256')]:require(read(evidence[group][1]['path'])[key]==evidence[group][0]['sha256'],'independent report binding '+group)
    require(read(evidence['M'][1]['path'])['raw_summary_sha256']==read(evidence['M'][0]['path'])['raw_summary_sha256'],'accepted mechanisms raw/corrected provenance')
    require(read(evidence['R'][1]['path'])['corrected_summary_sha256']==evidence['R'][0]['sha256'],'accepted source/query report binding')
    for group in ('L','S','M','G','E','R','H'):require(read(evidence[group][-1]['path'])['status'].startswith('passed'),'actual accepted experimental group '+group)
    sources=['benchmarks/v5/paper_multiscale/model.py','benchmarks/v4/phase_3_cell_region/src/model.py','benchmarks/v4/phase_5_prompt_encoder/src/model.py','benchmarks/v4/phase_6_mask_decoder/src/model.py','benchmarks/v5/fair_comparison/metrics.py']
    evidence['C']=[dict(path=str(Path(p).resolve()),sha256=sha(p),status='static_source_audit_only') for p in sources]
    require('each prompt set must contain at least one valid token' in Path(sources[2]).read_text(),'empty-negative restriction directly visible')
    require('nn.MultiheadAttention' in Path(sources[3]).read_text(),'dense masked attention cannot be claimed O(E) kernel')
    evidence['P']=[dict(path=str(a.manuscripts),sha256=sha(a.manuscripts),status=fingerprints['status'])]
    registry=read(a.registry);runtime=[]
    for job in registry['active_jobs']:
        path=Path(job['output'])/'status.json'
        try:observed=read(path) if path.exists() else None
        except (OSError,json.JSONDecodeError) as e:observed=dict(status='transient_progress_observation_unavailable',exception_type=type(e).__name__)
        runtime.append(dict(kind=job['kind'],worker_pid=job['worker_pid'],alive=Path('/proc/'+str(job['worker_pid'])).exists(),output=job['output'],observed_status=observed))
    (a.output/'runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n');evidence['RUN']=[dict(path=str(a.output/'runtime.json'),sha256=sha(a.output/'runtime.json'),status='live_observation_not_quality_acceptance')]
    rows=[]
    for authority,items in [('human',old['human_challenges']),('AI self-check',old['AI_self_checks'])]:
        for item in items:
            refs,state,response,gap=UPDATES[item['id']];ids=refs.split();require(all(i in evidence for i in ids),'all linked accepted evidence groups present')
            rows.append(dict(id=item['id'],note_id=item['note_id'],authority=authority,challenge=item['challenge'],current_evidence=state,supported_response=response,remaining=gap,evidence_ids=ids,manuscript_edit_applied=False,submitted_version_bound=False,resolution_status='not_declared_closed',paper_location='Local fingerprints/page/line evidence where available; proposed revision, no uploaded revision asserted'))
    require(len(rows)==25 and len({r['id'] for r in rows})==25 and {r['id'] for r in rows}==set(UPDATES),'all11 human and14 AI concerns without omissions')
    seed=read(evidence['S'][0]['path']);generic=read(evidence['G'][0]['path']);mechanisms=read(evidence['M'][0]['path']);external=read(evidence['E'][0]['path'])
    numbers=dict(local_three_seed_primary={m:v['3+2']['dice'] for m,v in seed['all_seed_summary'].items()},generic_primary={m:v['3+2']['dice']['mean'] for m,v in generic['metrics'].items()},generic_XCell_contrasts=[c for c in generic['contrasts'] if c['budget']=='3+2' and c['metric']=='dice' and c['b']=='fewclick_m3d_xcell'],mechanisms_primary=[c for c in mechanisms['accepted_contrasts'] if c['budget']=='3+2'],external_primary={m:v['3+2']['macro_pair_dice'] for m,v in external['metrics'].items()})
    for name,value in [('evidence.json',evidence),('challenges.json',rows),('accepted_numbers.json',numbers)]:
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    text=['# 审稿逐条证据更新（内部，不是已提交回复）','','11个人类challenge与14项AI自查全部保留。所有新实验属于重新设计的M3D/Dense/新训练协议；不是原140WSI或投稿F0复现。所有manuscript_edit_applied为false；G9仍preparing。','', '| 编号 | 质疑 | 当前实际证据 | 可以回应什么 | 仍缺什么 |','|---|---|---|---|---|']
    for r in rows:text.append('|'+ '|'.join([r['id'],r['challenge'],r['current_evidence'],r['supported_response'],r['remaining']])+'|')
    text+=['','数值及独立接受SHA见accepted_numbers.json/evidence.json。新的WSI未完成结果不进入数值表；运行状态仅在runtime.json。','', '本地稿件事实：main TeX391行/submission PDF7页的约.70 Dice与supp PDF4页的.6968 unprompted polygon recall不一致；all-TLS Dice为.5229±.2270。这是本地文本核对，不是原预测重算。local PDF/TeX指纹和页/行证据在上级local_manuscript_fingerprints.json，公开PDF匿名请求403，上传身份未绑定；未使用账号或cookie。','', '优先动作：完成既定主30预测/组件/距离/BF1/WSI统计；保持脂肪正常监督及原留出失败。随后绑定原提交/PDF、旧两manifest/5-of30与实际模型版本，按已支持范围改Methods/结果/局限。现有细胞/多尺度差值不能保证因果优势；同网络预训练目标、TLS难阴性与标注完整性仍是独立缺口。']
    (a.output/'report.md').write_text('\n'.join(text)+'\n')
    # Re-read every generated row and evidence pointer, never infer manuscript edits.
    check=read(a.output/'challenges.json');require(sum(r['authority']=='human' for r in check)==11 and sum(r['authority']=='AI self-check' for r in check)==14 and all(not r['manuscript_edit_applied'] and not r['submitted_version_bound'] for r in check),'coverage and claim-to-artifact limits')
    for group in read(a.output/'evidence.json').values():
        for p in group:require(sha(p['path'])==p['sha256'],'final all evidence pointer replay')
    for p in fingerprints['files']:require(sha(p['path'])==p['sha256'],'historical paper files remain read-only')
    summary=dict(status='accepted_reviewer_evidence_refresh_not_manuscript_resolution',worker_pid=os.getpid(),source_sha256=sha(__file__),human_challenges=11,AI_self_checks=14,review_snapshot_sha256=sha(a.reviews),local_manuscript_fingerprints_sha256=sha(a.manuscripts),artifacts={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},paper_modified=False,uploaded_version_bound=False,remaining='Main30 complete results/acceptance, reviewer evidence gaps and actual manuscript/submitted-version reconciliation; not overall goal completion')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('reviews','manuscripts','registry','output'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
