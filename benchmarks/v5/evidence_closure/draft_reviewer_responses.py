"""Build an internal response draft from accepted evidence, with no paper edits."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    board = a.root / 'reviewer_refresh_1028_151200/traceability_1028_151800'
    old = read(board / 'summary.json')
    require(old['status'] == 'accepted_reviewer_evidence_refresh_not_manuscript_resolution', 'accepted evidence board')
    for name, digest in old['artifacts'].items():
        require(sha(board / name) == digest, 'board artifact binding: ' + name)
    evidence = read(board / 'evidence.json')
    for group in evidence.values():
        for item in group:
            require(sha(item['path']) == item['sha256'], 'original evidence binding: ' + item['path'])
    reviews_path = a.root / 'reviewer_evidence_1025_163705/reviews.json'
    reviews = read(reviews_path)
    require(sha(reviews_path) == old['review_snapshot_sha256'], 'same review extraction')
    require(sha(reviews['source']) == reviews['source_sha256'], 'unchanged user-provided review HTML')
    notes = {n['note_id']: n for n in reviews['notes']}
    for n in notes.values():
        text = Path(n['saved_text']).read_text().removesuffix('\n')
        require(hashlib.sha256(text.encode()).hexdigest() == n['text_sha256'], 'verbatim review identity')
    numbers = read(board / 'accepted_numbers.json')
    seed = numbers['local_three_seed_primary']
    local = '；'.join(f"{m}: {seed[m]['mean']:.6f}±{seed[m]['sample_std']:.6f}" for m in
                     ('sam_ft', 'wsi_sam_ft', 'fewclick_m3d_image', 'fewclick_m3d_xcell'))
    contrast = {c['a']: c for c in numbers['generic_XCell_contrasts']}
    delta = contrast['generic_he_white64']
    cell = contrast['fewclick_m3d_image']
    dense = next(c for c in numbers['mechanisms_primary'] if c['label'] == 'dense_minus_region_XCell')
    external = numbers['external_primary']
    wsi = a.root / 'wsi_global_1026_010800'
    closed_path = wsi / 'closed12_Dice_BF1_accepted_report_1028_170100/summary.json'
    closed = read(closed_path)
    require(closed['status'] == 'descriptive_accepted_closed12_Dice_BF1_report', 'accepted descriptive closed report')
    native = wsi / 'closed12_raster_acceptance_1027_221200/summary.json'
    boundary = wsi / 'WSI_boundary_closed12_complete_1028_040500/summary.json'
    require(sha(native) == closed['native_acceptance_sha256'] and sha(boundary) == closed['boundary_summary_sha256'], 'closed native/boundary receipts')
    require(read(native)['status'].startswith('accepted_') and read(boundary)['status'] == 'accepted_complete_WSI_physical_BF1_rows', 'closed receipt statuses')
    efficiency_path = wsi / 'fullcell_efficiency_accepted_1028_143000/summary.json'
    efficiency = read(efficiency_path)
    require(efficiency['status'] == 'accepted_qualified_single_WSI_fullcell_efficiency_evidence', 'accepted measured pipeline costs')
    online = {r['budget']: r['wall_seconds'] for r in efficiency['online']}
    qualification = wsi / 'reference_label_provenance_1028_231445/evidence.json'
    q = read(qualification)
    require(q['status'] == 'provided_label_provenance_pending_user_clarification_not_model_acceptance', 'explicit pending label provenance')
    # This version requires pending provenance. A later clarification requires a new draft version.
    registry_bytes = a.registry.read_bytes()
    registry = json.loads(registry_bytes)
    registry_snapshot = a.output / 'registry_snapshot.json'
    registry_snapshot.write_bytes(registry_bytes)
    require(registry['provenance_clarifications']['main30_tissue_reference_labels']['status'] == 'pending_user_clarification', 'draft qualification still current')
    ce = closed['all330']['WSI_equal']
    matched = closed['supported276']['WSI_equal']
    responses = {
        'ZVeE-1': (
            '同队列监督训练与公共零样本基线的差异确实不能单独证明架构优势。新增受控协议使用相同新增标签池、40k实际任务和开发集选模规则，分别评估公共权重、适配权重和提供类别ID/全类别CE的闭集参考。'
            f'局部本图交互3+2预算的三seed均值±样本标准差为：{local}。当前结果不支持FewClick在该局部任务优于适配SAM。'
            '这一补充使用重新设计的数据/训练和M3D版本，并非原投稿140张训练集及旧模型的等价复现；整片远距任务另表，不能用局部成绩代替。',
            '实验：适配公平性、本图交互表、闭集参考表；AUTHOR_INPUT_NEEDED：原投稿模型和140张训练清单绑定。'),
        'ZVeE-2': (
            '当前主实验评估已知训练标签条件下，通过少量支持点击指定组织并检索远处区域的能力。12标签中脂肪正常参与训练；已完成的类别留出压力测试及其失败结果另列，不以该设置代替常规监督使用。'
            'TLS与非TLS淋巴聚集的语义区分、完整标注和难阴性尚需确认，因此现有证据不足以证明任意未见组织分割。'
            '本地稿件还存在约0.70被称为Dice的指标不一致：补充材料对应0.6968未提示TLS多边形召回，另报all-TLS Dice为0.5229±0.2270；这些是文本核对，未完成旧预测重算。',
            '任务定义、泛化实验、TLS指标和局限；AUTHOR_INPUT_NEEDED：TLS标注/难阴性与原预测指标重算。'),
        'ZVeE-3': (
            '新协议在看模型结果前固定全部30张WSI，保留来源/病例记录和空间分块政策，审计最大4x输入足迹。严格远距域包含20208个查询窗口；含支持带的28452窗口完整有效域作为独立应用诊断。'
            f'目前仅闭集12类CE、类别ID、无点击参考已有完整原生输出和独立Dice/BF1验收：全部330组织—WSI对的WSI等权Dice {ce["dice"]:.6f}、BF1@2µm {ce["bf1_um2"]:.6f}；'
            f'固定支持可用276对子集为{matched["dice"]:.6f}/{matched["bf1_um2"]:.6f}。这不是FewClick或交互基线成绩，不能与未完成的方法结果比较。'
            '11组织在全部30张参考分割图均有像素，整片缺失目标误报指标无合格样本，记NA，另报非目标像素误报及ROI缺失目标实验。原5/30的选择原因仍待原始记录追溯。',
            '实验：WSI集合/支持选择/完整域与远距域表；AUTHOR_INPUT_NEEDED：旧5/30选择及完整交互结果。'),
        'ZVeE-4': (
            '已记录的作者说明为XCell使用内部数据训练，当前CRC/TIGER未参与其训练；上游公开CTransPath图像预训练单独披露。推理仅需H&E，但这一表征包含额外预训练信息，不能称所有方法总监督相同。'
            '正式出处或自研未公开状态、内部训练规模/模态/划分及原200张队列资料仍需作者补齐，不能编造引用。新主30参考图的提供方README仅称10X组织分割结果，人工生成或专家审核状态待确认；确认前，数值按与提供参考图的一致性解释，不能先称专家金标准。',
            '数据、预训练与可复现性；AUTHOR_INPUT_NEEDED：XCell出处/内部资料、原200清单、参考图审核来源。'),
        'wHd8-1': (
            '方法可按数据流组织为：三物理视野构建区域及细胞表示并离线缓存；支持点击生成任务表示；查询窗口图解码并输出像素概率，最后拼接整片结果。'
            '新M3D/Dense路径需与旧投稿模型分别命名，不能把实际forward差异称作架构图笔误。192个节点对应单查询窗口的三视野区域，而非整张WSI跨tile图；当前拓扑约束由dense masked attention执行，也不等同于O(E)稀疏核。'
            '下面的回复是拟修改说明，原Methods、公式与图尚未实际编辑。',
            '拟改Methods结构和架构图/公式；AUTHOR_INPUT_NEEDED：确认提交版本，随后核对实际forward逐项修订。'),
        'wHd8-2': (
            '基线获得GT定位与模型是否经过目标数据适配是两个独立因素；前者不能抵消后者。新增协议分开报告公共零适配、同新增标签池适配及闭集特权参考，并分开本图交互与支持—查询迁移。'
            '所有模拟点击均由固定支持GT生成；查询GT不参与推理、候选选择或阈值，阈值固定0.5。SAM/WSI-SAM/FewClick的视野数、编码分辨率、预训练、容量和可训练层仍不同，清洁迁移模型也未经过R任务监督更新；因此只称新增任务匹配，不称等总监督、等计算或独立架构因果比较。',
            '实验协议/比较权限表/局限；AUTHOR_INPUT_NEEDED：原投稿实验对应版本及选模暴露绑定。'),
        'wHd8-3': (
            f'已完成一张真实KFB WSI的含细胞完整计算与独立原生输出核对：{efficiency["tiles"]}窗口、{efficiency["views"]}视野，完整pipeline耗时{efficiency["pipeline_wall_seconds"]:.3f}秒（包含首次准备和两次整片输出）；缓存后1+1和3+2预算整片输出分别为{online["1+1"]:.3f}秒和{online["3+2"]:.3f}秒。'
            '需分列读取、细胞准备、编码、缓存写入、解码与拼接IO，单任务解码毫秒不能当整片响应。该结果来自共享GPU节点的固定CRC支持到TLS查询机械样例，不是30张平均、受控加速比或TLS语义精度证明；其TLS标注完整性和负例仍待确认。'
            '细胞数量是重叠视野中的模型检测实例发生次数，不能称唯一细胞数或人工真值计数。',
            '效率实验及适用范围；AUTHOR_INPUT_NEEDED：主30完整系统时间、硬件共享条件和语义标注确认。'),
        'wHd8-4': (
            '本地稿件将0.7522描述为完整matched集合，0.7915对应共同子集；应在表题和图注明确集合、样本分母、点击预算及聚合公式。'
            '885个出现次数与819个唯一项不能混用。旧episode两份清单和全部旧逐项分数尚未绑定重算，实际上传PDF也未绑定；因此目前只能说明本地文本的解释，不能断言原结果已复现或已完成正文更正。新协议成绩应独立列出。',
            '旧Table1/Fig4图注与指标定义；AUTHOR_INPUT_NEEDED：原manifest/逐项分数/提交PDF。'),
        'wHd8-5': (
            '已完成无细胞、形态/计数、XCell及普通H&E特征两种预设投影控制，并保留全部结果。'
            f'在该单seed开发协议，Dense相对区域头差值{dense["b_minus_a"]:.6f}，来源聚类95%区间{dense["paired_source_cluster_ci95"]}；但Dense还增加共同训练的细语义路径，不能仅归因于投影。'
            f'XCell相对无细胞差值{cell["b_minus_a"]:.6f}，区间{cell["ci95"]}；相对白化普通HE差值{delta["b_minus_a"]:.6f}，区间{delta["ci95"]}，均包含0。'
            '这些结果尚不足以支持稳定的分子预训练因果收益。同网络/同数据的预训练目标控制仍未完成；原历史J5目标监督及混合任务训练结果另列。',
            '消融/额外先验说明/贡献与局限；AUTHOR_INPUT_NEEDED：若保留分子因果主张，补同网络同数据目标控制。'),
        'HVPZ-1': (
            '新增IGNITE外部域比较使用可提供真实256/512/1024µm视野的10个中心视野、9组织类，对六种方法固定相同支持点击，外部数据不参与本轮训练或选模。'
            f'3+2局部Dice：SAM-FT {external["sam_ft"]:.6f}、WSI-SAM-FT {external["wsi_sam_ft"]:.6f}、FewClick无细胞 {external["fewclick_m3d_image"]:.6f}、XCell {external["fewclick_m3d_xcell"]:.6f}；当前FewClick较低。'
            '这里的10是中心视野/场景位置，不是10家临床机构。公共权重使用CPU float32、适配权重使用GPU bfloat16，计算精度不同应披露；该比较不是速度对照。样本规模有限，应限定为小规模外部域结果，不能支持广泛跨癌种或跨机构泛化。PUMA/OCELOT现有小ROI无法提供完整上下文，不以重复或填充图像伪造1024µm视野。',
            '外部验证及视野可用性；AUTHOR_INPUT_NEEDED：更多可用原生WSI/中心和癌种。'),
        'HVPZ-2': (
            '当前已完成MultiverSeg上下文对手的迁移比较，分别报告预测支持上下文和完整支持mask参考，并在837共同子集比较。完整支持mask的标注信息多于少量点击，必须分表披露，不能按等交互成本解释。'
            '该比较不能证明已覆盖最近一年方法；近期方法的文献真实性、代码/权重可得性及任务兼容性仍需逐项核验，尚未运行的方法不会写成已比较。',
            '相关工作/对手信息权限表/共同子集结果；AUTHOR_INPUT_NEEDED：近期方法核验及可执行比较。'),
    }
    efficiency_report = wsi / 'fullcell_efficiency_accepted_1028_143000/report.md'
    rows = read(board / 'challenges.json')
    human = [r for r in rows if r['authority'] == 'human']
    require(len(human) == len(responses) == 11 and set(responses) == {r['id'] for r in human}, 'all human challenges covered exactly once')
    ai = [r for r in rows if r['authority'] == 'AI self-check']
    require(len(ai) == 14, 'all AI checks kept separate')
    generated = []
    intro = ('# 审稿逐条回复草稿（内部，尚未提交）\n\n'
             '本稿覆盖三位人类审稿人的11项质疑；题目为质疑摘要，原文以各note保存文本为准。'
             '新增实验采用重新设计的M3D/Dense与新训练协议，不冒充原投稿架构/140张WSI复现。'
             '下面是可编辑回复建议；原论文、公式和图尚未改动，也未上传。AUTHOR_INPUT_NEEDED表示未闭合工作。\n\n'
             '参考图来源限定：主30组织PNG的人工生成/专家审核方式尚待确认，当前Dice/BF1仅按与提供参考分割图的一致性解释；不据文件夹名称推断自动伪标签。\n')
    blocks = [intro]
    for r in human:
        note = notes[r['note_id']]
        require(note['authority'] == 'human' and note['reviewer'] == r['id'].split('-')[0], 'exact reviewer-note match')
        response, location = responses[r['id']]
        item = dict(id=r['id'], note_id=r['note_id'], challenge_paraphrase=r['challenge'], original_review=note['saved_text'],
                    original_review_sha256=note['text_sha256'], response_zh=response, proposed_location=location,
                    evidence_ids=r['evidence_ids'], open_work=r['remaining'], manuscript_edit_applied=False,
                    submitted_version_bound=False, resolution_status='internal_draft_not_resolved')
        generated.append(item)
        blocks.append(f'\n## {r["id"]}：{r["challenge"]}\n\n原始审稿文本：[{note["note_id"]}]({note["saved_text"]})。\n\n'
                      f'拟回复：{response}\n\n拟修改位置及待补：{location}\n\n'
                      f'原证据板未闭合项：{r["remaining"]}\n\n证据组：{", ".join(r["evidence_ids"])}，路径/哈希见 evidence.json。\n')
    blocks.append('\n## AI自查附表（非人类审稿意见）\n\n| ID | 自查项 | 未闭合工作 |\n|---|---|---|')
    for r in ai:
        blocks.append(f'| {r["id"]} | {r["challenge"]} | {r["remaining"]} |')
    paths = [board/'summary.json', board/'accepted_numbers.json', reviews_path, closed_path, native, boundary,
             qualification, efficiency_path, efficiency_report, registry_snapshot]
    bound = dict(inherited_evidence=evidence, current_evidence=[dict(path=str(p), sha256=sha(p)) for p in paths])
    (a.output/'evidence.json').write_text(json.dumps(bound, ensure_ascii=False, indent=2)+'\n')
    (a.output/'responses.json').write_text(json.dumps(dict(human=generated, AI_self_checks=ai), ensure_ascii=False, indent=2)+'\n')
    (a.output/'responses.md').write_text('\n'.join(blocks)+'\n')
    final = read(a.output/'responses.json')
    require(len(final['human']) == 11 and len(final['AI_self_checks']) == 14, 'saved draft coverage')
    require(all(not r['manuscript_edit_applied'] and not r['submitted_version_bound'] for r in final['human']), 'no invented edits or submission')
    for item in bound['current_evidence']:
        require(sha(item['path']) == item['sha256'], 'final source evidence binding')
    summary = dict(status='validated_internal_reviewer_response_draft_not_resolution', worker_pid=os.getpid(),
                   observed_at=datetime.datetime.now().astimezone().isoformat(), source_sha256=sha(__file__),
                   human_challenges=11, AI_self_checks=14, manuscript_modified=False, submitted=False,
                   artifacts={p.name:sha(p) for p in a.output.iterdir() if p.is_file()},
                   remaining='Main30 interactive/fullcell quality and statistics, original submission identity, missing provenance and reviewer evidence; full objective active')
    (a.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'registry', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
