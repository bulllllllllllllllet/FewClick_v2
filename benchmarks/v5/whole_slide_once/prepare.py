"""Audit declared/actual train-val membership, freeze slides and existing support."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pyvips
from benchmarks.v5.wsi_quality_study.prepare import dump
from benchmarks.v5.encoder_research.continuous_data import sha

BASE = Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research')
COHORT = Path('/nfs-medical3/zyh/v4/phase1/data/cohort200_20260716_010810/cohort_manifest.parquet')
PATCHES = Path('/nfs-medical3/zyh/v4/phase1/data/tiled_patch_index_cohort200_20260716_013248/patch_index_10x_tiled_gt.parquet')
HIGHRES = Path('/nfs-medical3/zyh/fewclick_v2/multidomain_training/highres_1023_025438/catalog_1023_025438/tiles.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    r = parser.parse_args().root
    r.mkdir(exist_ok=False)
    cohort = pd.read_parquet(COHORT)
    patches = pd.read_parquet(PATCHES)
    declared_seen = set(cohort.loc[cohort['split'].isin(['train', 'val']), 'wsi_id'])
    index_seen = set(patches.loc[patches['split'].isin(['train', 'val']), 'wsi_id'])
    highres = pd.read_parquet(HIGHRES, columns=['dataset', 'split', 'space_key'])
    highres_seen = {str(v).split('|')[-1] for v in highres.loc[
        (highres['dataset'] == 'crc') & highres['split'].isin(['train', 'val']), 'space_key']}
    tc = json.loads(Path('/nfs-medical3/zyh/fewclick_v2/fair_comparison/protocol_1025_124208/tasks_1025_125400/contract.json').read_text())
    current_catalog = pd.read_parquet(Path(tc['catalog'])/'tiles.parquet', columns=['dataset', 'split', 'space_key'])
    catalog_seen = {str(v).split('|')[-1] for v in current_catalog.loc[
        (current_catalog['dataset'] == 'crc') & current_catalog['split'].isin(['train', 'val']), 'space_key']}
    train_path = BASE/'same_slide_training_1110_191022/training_consumption.jsonl'
    train = [json.loads(v) for v in train_path.read_text().splitlines()]
    actual_seen = {str(t[k]).split('|')[-1] for t in train for k in ['source_space', 'query_space']}
    dev_path = BASE/'wsi_same_slide_1108_013913/tasks.json'
    development = json.loads(dev_path.read_text())
    development_seen = {t['query_space'].split('|')[-1] for t in development}
    history = BASE/'paper_closeout_1109_090821/confirmation_native_metadata'
    support_rows = json.loads((history/'query_rows.json').read_text())
    old_plans = json.loads((history/'plans.json').read_text())
    pool = cohort.loc[cohort['split'] == 'test'].to_dict('records')
    pool.sort(key=lambda t: hashlib.sha256(('hg_mix_whole_wsi_v1|'+t['wsi_id']).encode()).hexdigest())
    sources = dict(cohort=sha(COHORT), original_patch_index=sha(PATCHES), highres=sha(HIGHRES),
                   current_catalog=sha(Path(tc['catalog'])/'tiles.parquet'), actual_training=sha(train_path),
                   development=sha(dev_path), support_plans=sha(history/'plans.json'), support_rows=sha(history/'query_rows.json'))
    audit = []
    cases = []
    for index, item in enumerate(pool):
        slide = item['wsi_id']
        flags = {name: slide in seen for name, seen in [
            ('original_declared_train_val', declared_seen), ('original_patch_train_val', index_seen),
            ('historical_highres_train_val', highres_seen), ('current_catalog_train_val', catalog_seen),
            ('actual_current_training', actual_seen), ('current_development', development_seen)]}
        image = pyvips.Image.new_from_file(item['wsi_path'], access='random')
        gt = pyvips.Image.new_from_file(item['gt_path'], access='random')
        assert (image.width, image.height) == (item['level0_width'], item['level0_height'])
        mx, my = 1000/float(image.xres), 1000/float(image.yres)
        assert .05 < mx < 2 and .05 < my < 2
        ax, ay = gt.width/image.width, gt.height/image.height
        # Replay the original index's GT-native window rounding for these slides.
        old = patches.loc[patches['wsi_id'] == slide].head(64)
        assert len(old), slide
        for t in old.to_dict('records'):
            assert [round(t['x_10x']/ax), round(t['y_10x']/ay), round(t['width_10x']/ax), round(t['height_10x']/ay)] == [
                t['x_level0'], t['y_level0'], t['width_level0'], t['height_level0']], slide
        row = dict(wsi_id=slide, patient_id=str(item['patient_id']), wsi_path=item['wsi_path'], gt_path=item['gt_path'],
                   width=image.width, height=image.height, gt_width=gt.width, gt_height=gt.height,
                   native_mpp=[mx, my], gt_mpp=[mx/ax, my/ay], gt_affine=[ax, ay, 0, 0],
                   original_split='test', membership=flags, eligible=not any(flags.values()),
                   original_index_windows_replayed=len(old), original_alignment_status=str(item['alignment_status']),
                   original_historical_AAAI_test_exposure=True, selected=index < 15, pilot=index < 3)
        audit.append(row)
        if not row['eligible']:
            continue
        by_target = {}
        for task in old_plans:
            if task['support_space'].split('|')[-1] != slide:
                continue
            by_target.setdefault(int(task['target']), []).append(task)
        selected_supports = []
        for target, tasks in sorted(by_target.items()):
            task = min(tasks, key=lambda t: hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest())
            source = support_rows[task['support_id']]
            geometry = json.loads(source['multiscale_json'])
            assert geometry['source_image'] == item['wsi_path']
            assert len(task['positive_xy']) == 3 and len(task['negative_xy']) == 2
            # Original confirmation metadata copied unused template cell/patient fields;
            # retain only true geometry and use authoritative cohort provenance.
            selected_supports.append(dict(target=target, support_id=task['support_id'], geometry=geometry,
                positive_xy=task['positive_xy'], negative_xy=task['negative_xy'],
                support_labels=source['labels'], support_labels_sha256=sha(source['labels']),
                click_seed=task['click_seed'], click_audit=task.get('click_audit'), patient_id=str(item['patient_id'])))
        assert selected_supports, slide
        # One fixed, existing support target for the complete-slide pilot; no quality search.
        pilot_target = next((s['target'] for s in selected_supports if s['target'] == [0, 1, 10][index % 3]), selected_supports[0]['target'])
        cases.append(dict(**row, supports=selected_supports, pilot_target=pilot_target))
    selected = [v for v in audit if v['selected']]
    assert len(selected) == 15 and all(v['eligible'] for v in selected), 'Selected slide overlaps a declared/actual training or validation pool; stop'
    dump(r/'unseen_audit.json', dict(status='accepted_no_membership_in_audited_train_val_pools', sources=sources,
         original_split_counts=cohort.groupby('split').size().to_dict(), candidates=len(audit), selected=15,
         eligible=sum(v['eligible'] for v in audit), rows=audit,
         limitations='No membership in audited target train/val datasets, not proof about unknown public pretraining; historical AAAI test exposure retained. Patient-disjointness not imposed.'))
    dump(r/'cases.json', cases)
    dump(r/'preparation.json', dict(status='prepared_support_and_provenance_before_inference', selected_slides=15,
         pilot_slides=3, formal_eligible_tissue_tasks=sum(len(v['supports']) for v in cases if v['selected']),
         cases_sha256=sha(r/'cases.json'), audit_sha256=sha(r/'unseen_audit.json'),
         support_selection='Existing frozen confirmation support plans, SHA ordering; no new query-GT/prediction selection',
         pilot_target_rule='prefer fixed0/1/10 by slide index; if unavailable first existing supported class',
         coordinate_provenance='64 original index windows per slide replay native/GT rounding; real model/coverage gate still required'))
    print(json.dumps(dict(eligible=sum(v['eligible'] for v in audit), selected=15,
          tasks=sum(len(v['supports']) for v in cases if v['selected']))), flush=True)


if __name__ == '__main__':
    main()
