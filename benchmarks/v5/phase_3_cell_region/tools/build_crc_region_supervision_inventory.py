#!/usr/bin/env python3
"""Build panel-specific CRC regional cell/protein evidence without making tissue masks."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.crc_region_supervision import (STRUCTURE_CLASSES,IMMUNE_CLASSES,structure_class,immune_class,aggregate_chunk,merge_aggregates,add_coverage)

CASES=("2017-23800D","2019-00538F","2022-39941E")

def panel_table(path, mapper, classes, mpp, grid):
 chunks=[]
 for frame in pd.read_csv(path,sep="\t",usecols=["x","y","class"],chunksize=200000): chunks.append(aggregate_chunk(frame,mapper,classes,mpp,grid))
 return add_coverage(merge_aggregates(chunks,classes),classes,"panel")

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--timestamp",required=True); p.add_argument("--root",type=Path,default=Path("/nfs-medical6/data/SRRSH/crc")); p.add_argument("--output-root",type=Path,default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3")); p.add_argument("--grid-um",type=float,default=1024.); p.add_argument("--mpp",type=float,default=.2457); a=p.parse_args()
 if not re.fullmatch(r"\d{4}_\d{6}",a.timestamp) or min(a.grid_um,a.mpp)<=0: raise ValueError("timestamp and positive grid/mpp required")
 if not a.output_root.resolve().is_relative_to(Path("/nfs-medical3/zyh/fewclick_v2")): raise ValueError("FewClick artifact root required")
 out=a.output_root/f"crc_region_supervision_inventory_{a.timestamp}"; out.mkdir(parents=True,exist_ok=False); base=a.root/"panel_2/registration_outputs/he_panel2_to_panel1/common_coordinate_annotations"; reports=[]
 for case in CASES:
  one=panel_table(base/case/"panel1_cells_in_panel1_he_level0.tsv",structure_class,STRUCTURE_CLASSES,a.mpp,a.grid_um).rename(columns={"panel_observed_cells":"panel1_observed_cells","panel_mapped_cells":"panel1_mapped_cells","panel_coverage":"panel1_coverage",**{f"panel_{x}_count":f"panel1_{x}_count" for x in STRUCTURE_CLASSES}})
  two=panel_table(base/case/"panel2_cells_in_panel1_he_level0.tsv",immune_class,IMMUNE_CLASSES,a.mpp,a.grid_um).rename(columns={"panel_observed_cells":"panel2_observed_cells","panel_mapped_cells":"panel2_mapped_cells","panel_coverage":"panel2_coverage",**{f"panel_{x}_count":f"panel2_{x}_count" for x in IMMUNE_CLASSES}})
  merged=one.merge(two,on=["grid_x","grid_y"],how="outer").fillna(0); merged.insert(0,"case_id",case); merged["spatial_group_id"]=[f"{case}:grid{x}:grid{y}" for x,y in zip(merged.grid_x,merged.grid_y)]; merged["coordinate_system"]="panel1_HE_level0_um"; merged["grid_um"]=a.grid_um; merged.to_csv(out/f"{case}_regions.csv",index=False)
  reports.append({"case_id":case,"regions":len(merged),"panel1_observed_cells":int(merged.panel1_observed_cells.sum()),"panel2_observed_cells":int(merged.panel2_observed_cells.sum())})
 audit={"schema":"crc_region_supervision_inventory_v1","grid_um":a.grid_um,"mpp":a.mpp,"cases":reports,"scope":"panel-specific observed cell composition; not complete tissue composition, tissue mask, or training split"}; (out/"audit.json").write_text(json.dumps(audit,indent=2)); print(json.dumps({"output":str(out),**audit}))
if __name__=="__main__": main()
