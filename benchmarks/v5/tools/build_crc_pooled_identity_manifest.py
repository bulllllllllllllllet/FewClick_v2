#!/usr/bin/env python3
"""Derive an explicitly non-independent pooled CRC patch split for internal probing."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file

def pooled_rows(rows, val_per_case):
    groups={}
    for row in rows: groups.setdefault(row["case_id"],[]).append(dict(row))
    if len(groups)!=3 or val_per_case < 1: raise ValueError("three cases and positive val_per_case required")
    output=[]
    for case, values in sorted(groups.items()):
        values=sorted(values,key=lambda x:x["sample_id"])
        if len(values)<=val_per_case: raise ValueError(f"{case} has no train patch")
        for index,row in enumerate(values):
            row["split"]="val" if index >= len(values)-val_per_case else "train"
            output.append(row)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source-manifest",type=Path,required=True); p.add_argument("--timestamp",required=True); p.add_argument("--val-per-case",type=int,default=2); p.add_argument("--output-root",type=Path,default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3")); a=p.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}",a.timestamp): raise ValueError("timestamp must be MMDD_HHMMSS")
    allowed=Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    if not a.source_manifest.resolve().is_relative_to(allowed) or not a.output_root.resolve().is_relative_to(allowed): raise ValueError("FewClick artifact paths required")
    src=json.loads(a.source_manifest.read_text()); rows=pooled_rows(src["samples"],a.val_per_case)
    for row in rows:
        path=(a.source_manifest.parent/row["path"]).resolve()
        if sha256_file(path)!=row["sha256"]: raise ValueError(f"source checksum mismatch: {row['sample_id']}")
        row["path"]=str(path)
    out=a.output_root/f"crc_pooled_identity_manifest_{a.timestamp}"; out.mkdir(parents=True,exist_ok=False)
    src["samples"]=rows; src["provenance"]={**src["provenance"],"split_protocol":"pooled_case_overlapping_patch_internal_v1","split_scope":"All three cases occur in train and val; this is internal patch validation, not patient-independent evaluation."}
    (out/"manifest.json").write_text(json.dumps(src,indent=2))
    summary={split:{case:sum(r["split"]==split and r["case_id"]==case for r in rows) for case in sorted({r["case_id"] for r in rows})} for split in ("train","val")}
    (out/"metadata.json").write_text(json.dumps({"status":"complete","source_manifest":str(a.source_manifest),"source_manifest_sha256":sha256_file(a.source_manifest),"manifest_sha256":sha256_file(out/"manifest.json"),"split_counts":summary,"patient_independent":False,"warning":"Do not compare this internal result to patient-held-out performance."},indent=2))
    print(json.dumps({"status":"complete","output":str(out),"split_counts":summary}),flush=True)
if __name__=="__main__": main()
