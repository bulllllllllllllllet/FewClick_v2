#!/usr/bin/env python3
"""Build a source-only, spatially diverse CRC H&E processing manifest for label quotas."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.crc_sampling import FINE_CLASSES,quota_patch_table

CASES=("2017-23800D","2019-00538F","2022-39941E")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration-root',type=Path,default=Path('/nfs-medical5/hwx/panel2_cell_level_registration'));p.add_argument('--timestamp',required=True);p.add_argument('--common-per-case',type=int,default=16667);p.add_argument('--max-dapi-cells',type=int,default=2000);p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));a=p.parse_args()
 if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp) or a.common_per_case<1 or a.max_dapi_cells<30:raise ValueError('valid timestamp and positive quotas required')
 if not a.output_root.resolve().is_relative_to(Path('/nfs-medical3/zyh/fewclick_v2')):raise ValueError('FewClick output only')
 out=a.output_root/f'crc_quota_manifest_{a.timestamp}';out.mkdir(parents=True,exist_ok=False); reports={}; rows=[]
 for case in CASES:
  folder='cell_annotations_mif_only' if case=='2019-00538F' else 'cell_annotations';table=a.registration_root/case/folder/f'{case}_registered_celltypes.tsv'
  targets={key:(0 if key=='Treg' else a.common_per_case) for key in FINE_CLASSES}
  first=quota_patch_table(table,targets,a.max_dapi_cells);targets['Treg']=first['source_class_counts']['Treg'];report=quota_patch_table(table,targets,a.max_dapi_cells);reports[case]=report
  for index,patch in enumerate(report['selected_patches']):rows.append({'case_id':case,'patch_name':patch,'shard':index%6})
  (out/f'{case}_selection.json').write_text(json.dumps(report,indent=2));print(json.dumps({'case':case,'patches':len(report['selected_patches']),'achieved':report['achieved_source_candidates']}),flush=True)
 (out/'patch_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));(out/'metadata.json').write_text(json.dumps({'status':'complete','target_common_per_case':a.common_per_case,'target_common_total':a.common_per_case*3,'treg':'all source candidates in eligible patches','max_dapi_cells':a.max_dapi_cells,'reports':reports,'shards':6,'scope':'source-only selection; no H&E detection, feature extraction, or label transfer has run'},indent=2));print(json.dumps({'status':'complete','output':str(out),'patches':len(rows)}),flush=True)
if __name__=='__main__':main()
