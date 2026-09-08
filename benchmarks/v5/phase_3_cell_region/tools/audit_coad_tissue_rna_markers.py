#!/usr/bin/env python3
"""Audit availability and expression of tissue/TLS marker programs in COAD Xenium."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(0,str(ROOT))
from run_xenium_rna_pilot import load_transcripts, normalized_counts
SETS={'epithelial':['EPCAM','KRT8','KRT18','KRT19','KRT20','MUC1','CEACAM5'],'stromal_ecm':['COL1A1','COL1A2','COL3A1','DCN','LUM','COL6A1','TAGLN'],'t_cell':['CD3D','CD3E','TRBC1','TRBC2','IL7R','LTB'],'b_plasma_tls':['MS4A1','CD79A','CD74','CD37','JCHAIN','MZB1','CXCL13','CCL19','CCL21'],'myeloid':['LYZ','LST1','TYROBP','C1QA','C1QB','FCER1G'],'muscle':['ACTA2','MYH11','TAGLN','DES'],'mucus':['MUC2','MUC5AC','AGR2','SPINK4'],'blood':['HBA1','HBA2','HBB','ALAS2'],'adipose':['ADIPOQ','PLIN1','FABP4','LPL']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--data-root',type=Path,default=Path('/nfs-medical3/fyx/PKU_DATA/Xenium/COAD'));p.add_argument('--summary',type=Path,default=Path('/nfs-medical3/zyh/fewclick_v2/rna_pilot/xenium_xcell_baseline_0831_111806/cell_summary.csv'));a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 cells=pd.read_csv(a.summary,dtype={'cell_id':str});tx=load_transcripts(a.data_root/'transcripts/transcripts.parquet',set(cells.cell_id),1000,2000,1500,20.0);counts=normalized_counts(tx,cells.cell_id.tolist(),30);rows=[]
 for name,markers in SETS.items():
  available=[g for g in markers if g in counts];score=counts[available].mean(1) if available else pd.Series(0.,index=counts.index);rows.append({'program':name,'requested_markers':'|'.join(markers),'available_markers':'|'.join(available),'coverage':len(available)/len(markers),'cells_nonzero_fraction':float((score>0).mean()),'score_mean':float(score.mean()),'score_std':float(score.std())})
 a.output_dir.mkdir(parents=True);pd.DataFrame(rows).to_csv(a.output_dir/'marker_program_audit.csv',index=False);meta={'cells':len(counts),'genes':len(counts.columns),'programs':rows,'acceptance':'program needs >=2 available markers and nonzero variation before it can be a target'};(a.output_dir/'metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps({'event':'complete','output':str(a.output_dir),'cells':len(counts),'programs':rows},ensure_ascii=False))
if __name__=='__main__':main()
