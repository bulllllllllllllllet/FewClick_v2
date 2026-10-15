"""Fixed test examples: real H&E, unchanged prompts, paired masks and GT-only zooms."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from . import module_source
from .boundary_data import load_baseline, make_loader, move_batch
from .dense_prompt_decoder import prompt_maps
from .continue_dense_boundary import DenseContextModel
from .point_boundary_detail import PointBoundaryDetail
from .multiscale_dense_decoder import MultiscaleDenseModel
from .train_dense_boundary import select_indices, points
from .probe_boundary_refinement import metrics, summarize, prompt_violations
from .boundary_refinement import dilate
from benchmarks.v4.phase_6_mask_decoder.src.evaluation import binary_boundary


ARTIFACT_ROOT = Path('/nfs-medical3/zyh/fewclick_v2')
BASELINE_ROOT = Path('/nfs-medical3/zyh/v4/phase6/formal_full_runs/phase6_joint_pixel_20260722_142455')
MULTISCALE_RUN = ARTIFACT_ROOT/'boundary_refinement/multiscale_1015_133617/pilot_1015_133617'
FINAL_CHECKPOINT = MULTISCALE_RUN/'full256_epoch_007.pth'
LABELS = {'j5': '原模型 J5', 'dense': '稠密解码', 'point': '边界监督细化', 'multiscale': '多尺度候选（最后轮）'}
CLASS_NAMES = ['肿瘤上皮', '肿瘤间质', '背景', '坏死', '正常腺体', '正常间质',
               '黏膜下层/浆膜', '肌层', '淋巴聚集', '黏液', '脂肪', '血液']


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def boundary_zoom_box(truth, valid, size=128, stride=32):
    """Choose a reproducible GT boundary window, never looking at predictions."""
    edge = binary_boundary(truth, valid) & valid & ~dilate(~valid, 4)
    pooled = F.avg_pool2d(edge[:, None].float(), size, stride=stride)[0, 0]
    positive = F.avg_pool2d((truth & valid)[:, None].float(), size, stride=stride)[0, 0] > 0
    negative = F.avg_pool2d((~truth & valid)[:, None].float(), size, stride=stride)[0, 0] > 0
    eligible = positive & negative
    if not eligible.any():
        raise ValueError('no zoom window contains both valid target and background')
    index = int(pooled.masked_fill(~eligible, -1).flatten().argmax())
    y, x = divmod(index, pooled.shape[1])
    return [x*stride, y*stride, size, size]


def rgb_image(sample):
    image = sample['image'][0].float().permute(1, 2, 0).cpu().numpy()
    return np.clip(image*np.array([.229, .224, .225])+np.array([.485, .456, .406]), 0, 1)


def metric_row(probability, sample, identity):
    row = {**identity, **metrics(probability, sample['truth'], sample['valid'])}
    row['prompt_violations'] = prompt_violations(probability, sample)
    row['credible_prompts'], row['credible_violations'] = points(probability, sample)
    return row


def configure_font():
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus'] = False


def paint(axes, sample, predictions, rows, zoom=None, rectangle=None):
    image = rgb_image(sample)
    truth = sample['truth'][0].cpu().numpy()
    valid = sample['valid'][0].cpu().numpy()
    height, width = truth.shape
    panels = [('image', None), ('gt', None), *predictions.items()]
    for ax, (key, probability) in zip(axes, panels):
        ax.imshow(image)
        ignore = np.zeros((height, width, 4)); ignore[~valid] = [.4, .4, .4, .7]
        ax.imshow(ignore)
        if key == 'gt':
            overlay = np.zeros((height, width, 4)); overlay[truth & valid] = [0, .8, .2, .25]
            ax.imshow(overlay)
        if key != 'image' and truth[valid].any() and (~truth[valid]).any():
            ax.contour(np.ma.array(truth.astype(float), mask=~valid), levels=[.5],
                       colors=['#00D43B'], linewidths=1.1, linestyles='dashed')
        if probability is not None:
            pred = probability >= .5
            if pred[valid].any() and (~pred[valid]).any():
                ax.contour(np.ma.array(pred.astype(float), mask=~valid), levels=[.5],
                           colors=['#FF8200'], linewidths=1.1)
            row = rows[key]; bf = row['bf1_2']
            ax.set_title(f"{LABELS[key]}\nDice {row['dice']:.3f} | BF1 {bf:.3f}" if bf is not None
                         else f"{LABELS[key]}\nDice {row['dice']:.3f} | BF1 NA", fontsize=10)
        else:
            ax.set_title('原图与原始提示' if key == 'image' else '目标组织标注', fontsize=10)
        for name, color, marker in [('positive', '#007BFF', '+'), ('negative', '#CB00DE', 'x')]:
            xy = sample[name+'_xy'][0][sample[name+'_mask'][0]].cpu().numpy()
            if len(xy):
                ax.scatter(xy[:, 0]*width-.5, xy[:, 1]*height-.5, s=42, c=color, marker=marker, linewidths=1.6)
        if rectangle is not None:
            x, y, w, h = rectangle
            ax.add_patch(Rectangle((x-.5, y-.5), w, h, fill=False, edgecolor='#007BFF', linewidth=1.1))
        if zoom is not None:
            x, y, w, h = zoom
            ax.set_xlim(x-.5, x+w-.5); ax.set_ylim(y+h-.5, y-.5)
        ax.axis('off')


def render_case(sample, probabilities, rows, case, destination):
    columns = 2+len(probabilities)
    fig, axes = plt.subplots(2, columns, figsize=(3.4*columns, 8.8), squeeze=False)
    paint(axes[0], sample, probabilities, rows, rectangle=case['zoom'])
    paint(axes[1], sample, probabilities, rows, zoom=case['zoom'])
    title = f"固定测试例 {case['case']+1:02d}：{CLASS_NAMES[case['target_class']]} | {case['patch_id']}"
    fig.suptitle(title+f"\n空间 test；旧划分来源 {case['historical_split']}；下排为同位置放大", fontsize=12)
    fig.text(.5, .013, '绿虚线=标注；橙实线=预测；蓝+=正提示；紫×=负提示；灰=忽略区。放大框仅由GT边界确定，未按提升幅度选择。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.015, right=.995, top=.87, bottom=.07, wspace=.08, hspace=.30)
    fig.savefig(destination, dpi=160); plt.close(fig)


def render_overviews(root, cases, probability_names):
    for page, first in enumerate(range(0, len(cases), 4), 1):
        subset = cases[first:first+4]
        fig, axes = plt.subplots(len(subset), 2+len(probability_names), figsize=(3.2*(2+len(probability_names)), 3.9*len(subset)+.7), squeeze=False)
        for axrow, case in zip(axes, subset):
            sample = torch.load(root/'cache'/f"case_{case['case']:02}.pth", map_location='cpu', weights_only=False)
            archive = np.load(root/'predictions'/f"case_{case['case']:02}.npz")
            probabilities = {k: archive[k] for k in probability_names}
            rows = {k: case['metrics'][k] for k in probability_names}
            paint(axrow, sample, probabilities, rows)
            axrow[0].set_title(f"{case['case']+1:02d} {CLASS_NAMES[case['target_class']]}\n原图及原始提示", fontsize=10)
        fig.suptitle('固定测试样本对比（全部保留，不按增益挑图） | 绿虚线=GT，橙实线=预测', fontsize=12)
        fig.subplots_adjust(left=.015, right=.995, top=.94, bottom=.02, wspace=.08, hspace=.30)
        fig.savefig(root/f'overview_{page:02}.png', dpi=145); plt.close(fig)


def save_report(root, cases, names, run):
    report = {'scope': 'fixed class-balanced test visualization subset, not full test evaluation; historical encoder exposure',
              'count': len(cases), 'test_used': True, 'models': {}, 'cases': cases}
    for name in names:
        rows = [c['metrics'][name] for c in cases]
        summary = summarize(rows)
        summary['credible_prompts'] = sum(r['credible_prompts'] for r in rows)
        summary['credible_violations'] = sum(r['credible_violations'] for r in rows)
        report['models'][name] = summary
    (root/'report.json').write_text(json.dumps(report, indent=2))
    fields = ['case', 'patch_id', 'target_class', 'historical_split', 'model', 'dice', 'bf1_2', 'boundary_band_iou_2', 'credible_violations']
    with (root/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for case in cases:
            for name in names:
                writer.writerow({**{k: case[k] for k in fields[:4]}, 'model': name,
                                 **{k: case['metrics'][name][k] for k in fields[5:]}})
    content = ['<!doctype html><meta charset="utf-8"><title>测试图块边界对比</title>',
               '<style>body{max-width:1800px;margin:30px auto;font-family:sans-serif}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>',
               '<h1>固定测试图块边界对比</h1><p>每类预先抽样一张；全部结果保留。仅展示子集，不代表全测试集；旧encoder存在历史区域暴露。模型由训练方案预先指定，不按测试指标选模。</p>',
               '<p>绿虚线：GT；橙实线：预测；蓝+：正提示；紫×：负提示。放大框仅由GT确定。<a href="metrics.csv">逐例指标CSV</a> · <a href="report.json">完整报告</a></p>',
               '<table><tr><th>模型</th><th>Dice</th><th>BF1@2</th></tr>']
    for name, row in report['models'].items():
        bf = f"{row['bf1_2']:.4f}" if row['bf1_2'] is not None else 'NA'
        content.append(f"<tr><td>{LABELS[name]}</td><td>{row['dice']:.4f}</td><td>{bf}</td></tr>")
    content.append('</table>')
    for case in cases:
        image = f"case_{case['case']:02}.png"
        content.append(f"<h2>{case['case']+1:02d} {CLASS_NAMES[case['target_class']]}</h2><a href='{image}'><img loading='lazy' src='{image}'></a>")
    (root/'index.html').write_text('\n'.join(content))
    (root/'run.json').write_text(json.dumps(run, indent=2))
    (root/'status.json').write_text(json.dumps({'status': 'complete', 'worker_pid': os.getpid(), 'count': len(cases)}))
    print('COMPLETE', json.dumps(report['models']), flush=True)


def append_cases(model, prepared, output, device):
    cases = json.loads((prepared/'report.json').read_text())['cases']
    for case in cases:
        i = case['case']; source = prepared/'cache'/f'case_{i:02}.pth'
        (output/'cache'/source.name).symlink_to(source)
        sample = torch.load(source, map_location=device, weights_only=False)
        with np.load(prepared/'predictions'/f'case_{i:02}.npz') as old:
            probabilities = {name: old[name] for name in ['j5', 'dense', 'point']}
        with torch.no_grad():
            p = model(sample)['pixel_probability']
            altered = {**sample, 'truth': ~sample['truth'], 'valid': ~sample['valid']}
            if not torch.equal(p, model(altered)['pixel_probability']):
                raise RuntimeError('GT-dependent inference')
        probabilities['multiscale'] = p[0].cpu().numpy()
        case['metrics']['multiscale'] = metric_row(p, sample, case['metrics']['dense'])
        np.savez_compressed(output/'predictions'/f'case_{i:02}.npz', **probabilities)
        render_case(sample, probabilities, case['metrics'], case, output/f'case_{i:02}.png')
        (output/'status.json').write_text(json.dumps({'status': 'rendering', 'completed': i+1, 'worker_pid': os.getpid()}))
        print('CASE', i, flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--multiscale-checkpoint', type=Path)
    args = parser.parse_args()
    if ARTIFACT_ROOT not in args.output.resolve().parents or not 1 <= args.count <= 12:
        raise ValueError('artifact root/count')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'predictions').mkdir(); (args.output/'cache').mkdir()
    torch.set_num_threads(4); configure_font(); device = torch.device('cuda:0')
    coarse_path = ARTIFACT_ROOT/'boundary_refinement/next_stage_1014_200033/matched96k_1014_221710/dense128_epoch_031.pth'
    point_path = ARTIFACT_ROOT/'boundary_refinement/point_objective_1015_125213/pilot_1015_125213/global_epoch_007.pth'
    coarse_state = torch.load(coarse_path, map_location='cpu', weights_only=False)['model']
    run = {'worker_pid': os.getpid(), 'test_used': True, 'seed': 20261015,
           'planned_multiscale_checkpoint': str(FINAL_CHECKPOINT), 'selection_policy': 'one unique test patch per class before prediction; no gain-based replacement',
           'scope': 'fixed display subset; historical encoder exposure; simulated fixed prompts',
           'source_sha256': {str(module_source(m)): file_digest(module_source(m)) for m in
                             ['visualize_test_boundary', 'dense_prompt_decoder', 'point_boundary_detail', 'multiscale_dense_decoder']}}
    if args.prepared:
        if args.multiscale_checkpoint != FINAL_CHECKPOINT:
            raise ValueError('only the predeclared final checkpoint may be appended')
        if json.loads((args.prepared/'status.json').read_text())['status'] != 'complete' or json.loads((MULTISCALE_RUN/'status.json').read_text())['status'] != 'complete':
            raise RuntimeError('prepared display/formal training incomplete')
        checkpoint = torch.load(FINAL_CHECKPOINT, map_location='cpu', weights_only=False)
        if checkpoint['mode'] != 'full256' or checkpoint['epoch'] != 7:
            raise RuntimeError('final checkpoint identity')
        model = MultiscaleDenseModel(coarse_state, 'full256').to(device)
        model.load_state_dict(checkpoint['model'], strict=True); model.eval().requires_grad_(False)
        run.update(prepared=str(args.prepared), checkpoint=str(FINAL_CHECKPOINT), checkpoint_sha256=file_digest(FINAL_CHECKPOINT))
        cases = append_cases(model, args.prepared, args.output, device)
        names = ['j5', 'dense', 'point', 'multiscale']
    else:
        metadata = json.loads((BASELINE_ROOT/'run_metadata.json').read_text())
        data = argparse.Namespace(spatial_data=ARTIFACT_ROOT/'extensible_prompt/tissue_spatial_split_0922_001235',
                                  routing=ARTIFACT_ROOT/'iterative_decoder/spatial_routing_0923_154545',
                                  seed=20261015, train_samples=args.count, val_samples=args.count)
        loader = make_loader(data, metadata, 'test'); dataset = loader.dataset.dataset
        indices = select_indices(dataset, args.count, 20261015)
        manifest = pd.read_parquet(data.spatial_data/'spatial_manifest.parquet').set_index('patch_id')
        selection = []
        for i, index in enumerate(indices):
            patch = str(dataset.episodes.patch_ids[index])
            if manifest.loc[patch, 'split'] != 'test': raise RuntimeError('not a test patch')
            selection.append({'case': i, 'episode_index': index, 'patch_id': patch,
                              'target_class': int(dataset.episodes.target_classes[index]),
                              'historical_split': str(manifest.loc[patch, 'historical_split'])})
        (args.output/'selection.json').write_text(json.dumps(selection, indent=2))
        run.update(checkpoints={'j5': str(BASELINE_ROOT/'checkpoint_epoch_004.pth'), 'dense': str(coarse_path), 'point': str(point_path)})
        run['checkpoint_sha256'] = {name: file_digest(path) for name, path in run['checkpoints'].items()}
        (args.output/'run.json').write_text(json.dumps(run, indent=2))
        baseline = load_baseline(metadata, BASELINE_ROOT/'checkpoint_epoch_004.pth', device)
        dense = DenseContextModel().to(device); dense.load_state_dict(coarse_state, strict=True); dense.eval().requires_grad_(False)
        point = PointBoundaryDetail(coarse_state).to(device)
        point.load_state_dict(torch.load(point_path, map_location=device, weights_only=False)['model'], strict=True); point.eval().requires_grad_(False)
        captured = {}
        def capture(name):
            def hook(module, inputs, output): captured[name] = output
            return hook
        hooks = [baseline.phase2.embedding.register_forward_hook(capture('deep')),
                 baseline.phase2.backbone.relu.register_forward_hook(capture('stem')),
                 baseline.phase2.backbone.layer1.register_forward_hook(capture('native_shallow')),
                 baseline.phase2.backbone.layer2.register_forward_hook(capture('middle'))]
        selected_loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=0, collate_fn=loader.collate_fn)
        cases = []
        for i, batch in enumerate(selected_loader):
            batch = move_batch(batch, device); case = dict(selection[i])
            if str(batch['patch_id'][0]) != case['patch_id']: raise RuntimeError('sample selection identity')
            with torch.no_grad():
                out = baseline(batch)
                features = {k: v.detach().half() for k, v in captured.items()}
                features['shallow'] = captured['native_shallow'].reshape(1, 32, 8, 128, 128).mean(2).half()
                sample = {**features, 'image': batch['image'], 'semantic': F.interpolate(out['semantic_logits'], size=(64, 64), mode='bilinear', align_corners=False).half(),
                          'task': out['task_token'].float(), 'base': out['pixel_probability'].float(), 'prompt_maps': prompt_maps(batch).half(),
                          'truth': batch['pixel_gt'] == batch['target_class'][:, None, None], 'valid': batch['pixel_gt'] != 255}
                sample.update({k: batch[k] for k in ['positive_xy', 'negative_xy', 'positive_mask', 'negative_mask']})
                if not (sample['truth'] & sample['valid']).any() or not (~sample['truth'] & sample['valid']).any(): raise ValueError('binary test support')
                if not all(torch.isfinite(v).all() for v in sample.values()): raise FloatingPointError('test cache')
                predictions = {'j5': sample['base'], 'dense': dense(sample)['pixel_probability'], 'point': point(sample)['pixel_probability']}
                if i == 0:
                    altered = {**batch, 'pixel_gt': torch.where(batch['pixel_gt'] == 255, batch['pixel_gt'], (batch['pixel_gt']+1) % 12)}
                    if not torch.equal(predictions['j5'], baseline(altered)['pixel_probability']): raise RuntimeError('baseline query-GT-dependent prediction')
            case['zoom'] = boundary_zoom_box(sample['truth'], sample['valid'])
            case['prompts'] = {name: sample[name+'_xy'][0][sample[name+'_mask'][0]].cpu().tolist() for name in ['positive', 'negative']}
            case['metrics'] = {name: metric_row(p, sample, case) for name, p in predictions.items()}
            probabilities = {name: p[0].cpu().numpy() for name, p in predictions.items()}
            torch.save({k: v.cpu() for k, v in sample.items()}, args.output/'cache'/f'case_{i:02}.pth')
            np.savez_compressed(args.output/'predictions'/f'case_{i:02}.npz', **probabilities)
            render_case(sample, probabilities, case['metrics'], case, args.output/f'case_{i:02}.png')
            cases.append(case)
            (args.output/'status.json').write_text(json.dumps({'status': 'rendering', 'completed': i+1, 'worker_pid': os.getpid()}))
            print('CASE', i, case['patch_id'], flush=True)
        for hook in hooks: hook.remove()
        names = ['j5', 'dense', 'point']
    render_overviews(args.output, cases, names)
    save_report(args.output, cases, names, run)


if __name__ == '__main__':
    main()
