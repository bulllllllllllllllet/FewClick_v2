#!/usr/bin/env python3
"""Evaluate a spatial-context identity checkpoint on an explicit manifest split."""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import SemanticDataset,sha256_file
from benchmarks.v5.phase_3_cell_region.src.spatial_identity import SpatialContextIdentity
from benchmarks.v5.phase_3_cell_region.tools.train_crc_spatial_identity import evaluate
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--timestamp',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--output-root',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3'));a=p.parse_args()
 if not re.fullmatch(r'\d{4}_\d{6}',a.timestamp):raise ValueError('timestamp must be MMDD_HHMMSS')
 data=SemanticDataset(a.manifest,a.split);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if saved['manifest_sha256']!=data.manifest_hash:raise ValueError('checkpoint/manifest mismatch')
 model=SpatialContextIdentity(len(saved['identity_classes']),neighbors=saved['config']['neighbors']).to(a.device);model.load_state_dict(saved['model'],strict=True);metrics=evaluate(model,data,torch.device(a.device));out=a.output_root/f'evaluate_spatial_{a.split}_{a.timestamp}';out.mkdir(parents=True,exist_ok=False)
 (out/'complete.json').write_text(json.dumps({'checkpoint_sha256':sha256_file(a.checkpoint),'manifest_sha256':data.manifest_hash,'split':a.split,'metrics':metrics},indent=2));print(json.dumps({'status':'complete','output':str(out),**metrics}),flush=True)
if __name__=='__main__':main()
