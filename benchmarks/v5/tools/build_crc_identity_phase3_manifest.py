#!/usr/bin/env python3
"""Build Phase-3 identity contracts from doctor-reviewed CRC H&E cell caches."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from benchmarks.v4.phase_1_multiscale.src.common import load_config
from benchmarks.v4.phase_2_region_encoder.src.model import DeepRegionEncoder
from benchmarks.v5.phase_3_cell_region.src.semantic_contract import sha256_file

CLASSES=["CD4_T","CD8_T","Treg","B_cell","CD163_Macrophage"]
SPLITS={"2017-23800D":"train","2019-00538F":"val","2022-39941E":"test"}

def cells_from_cache(raw: dict[str,np.ndarray]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    n=len(raw["he_cell_ids"]); h,w=raw["he_mask"].shape
    xy=raw["he_xy"].astype(np.float32)/np.array([w-1,h-1],np.float32)
    if not ((xy>=0).all() and (xy<=1).all()): raise ValueError("cell centers outside H&E patch")
    area=np.bincount(raw["he_mask"].ravel(),minlength=int(raw["he_mask"].max())+1)[raw["he_cell_ids"]]
    values=np.zeros((n,74),np.float32); values[:,:2]=xy; values[:,2]=np.log1p(area); values[:,3]=1.; values[:,10:]=raw["reg_features"]
    valid=raw["identity_verified"].astype(bool)
    allowed=np.zeros((n,len(CLASSES)),bool); allowed[np.arange(n)[valid],raw["candidate_class"][valid]]=True
    return values,valid,allowed

def tensor_image(path: Path) -> torch.Tensor:
    image=np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)/255.
    return torch.from_numpy(image.transpose(2,0,1)).sub(torch.tensor([.485,.456,.406])[:,None,None]).div(torch.tensor([.229,.224,.225])[:,None,None])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,required=True); p.add_argument("--timestamp",required=True)
    p.add_argument("--output-root",type=Path,default=Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3")); p.add_argument("--device",default="cuda:1")
    p.add_argument("--max-per-case",type=int,default=0,help="positive count for representative smoke, zero for all")
    p.add_argument("--phase2-config",type=Path,default=Path("/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/config_snapshot.json"))
    p.add_argument("--phase2-checkpoint",type=Path,default=Path("/nfs-medical3/zyh/v4/phase2/runs/phase_2_region_encoder_train_20260717_161240/checkpoint_epoch_028.pth"))
    a=p.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}",a.timestamp): raise ValueError("timestamp must be MMDD_HHMMSS")
    allowed_root=Path("/nfs-medical3/zyh/fewclick_v2").resolve()
    if not a.cache_dir.resolve().is_relative_to(allowed_root) or not a.output_root.resolve().is_relative_to(allowed_root): raise ValueError("FewClick artifacts only")
    meta=json.loads((a.cache_dir/"metadata.json").read_text())
    if meta.get("schema")!="crc_panel2_expert_reviewed_mif_transferred_v1" or meta.get("identity_classes")!=CLASSES or not meta.get("source_expert_reviewed"): raise ValueError("expert-reviewed CRC cache required")
    out=a.output_root/f"crc_identity_phase3_manifest_{a.timestamp}"; out.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device); cfg=load_config(a.phase2_config); phase2=DeepRegionEncoder(cfg,len(cfg["data"]["class_map"]),64,256).to(device)
    phase2.load_state_dict(torch.load(a.phase2_checkpoint,map_location="cpu",weights_only=False)["model"],strict=True); phase2.eval().requires_grad_(False)
    records=[]; seen={key:0 for key in SPLITS}
    for row in meta["records"]:
        case=row["case_id"]
        if a.max_per_case and seen[case]>=a.max_per_case: continue
        source=a.cache_dir/row["path"]
        if sha256_file(source)!=row["sha256"]: raise ValueError(f"checksum mismatch: {source.name}")
        with np.load(source,allow_pickle=False) as src: raw={k:src[k].copy() for k in src.files}
        cells,identity_valid,identity_allowed=cells_from_cache(raw)
        image=tensor_image(Path(row["he_path"])).unsqueeze(0).to(device)
        with torch.no_grad(): result=phase2(image,return_full_assignment=False,return_tokens=True)
        grid=torch.from_numpy(cells[:,:2]).to(device).mul(2).sub(1).view(1,1,-1,2)
        mass=F.grid_sample(result["assignment_low"],grid,align_corners=False,padding_mode="border").squeeze(2).cpu().numpy()[0]
        token=result["region_tokens"].cpu().numpy()[0]
        if not (np.isfinite(token).all() and np.isfinite(mass).all() and np.allclose(mass.sum(0),1.,atol=1e-5)): raise ValueError("invalid phase2 output")
        name=f"{case}_{row['patch_name']}_phase3.npz"
        # No physical MPP was supplied for these registered patches: area is a positive placeholder and all area-dependent targets are invalid.
        np.savez_compressed(out/name,region_tokens=token.astype(np.float32),cells=cells,cell_valid=np.ones(len(cells),bool),assignment_at_cells=mass.astype(np.float32),region_area_um2=np.ones(64,np.float32),identity_allowed=identity_allowed,identity_valid=identity_valid,tissue_target=np.zeros((64,2),np.float32),tissue_valid=np.zeros(64,bool),composition_target=np.zeros((64,5),np.float32),composition_valid=np.zeros(64,bool),cell_ids=raw["he_cell_ids"])
        record={"sample_id":name.removesuffix(".npz"),"case_id":case,"organ":"colon","split":SPLITS[case],"path":name,"sha256":sha256_file(out/name),"identity_labels":int(identity_valid.sum())}
        records.append(record); seen[case]+=1; print(json.dumps(record),flush=True)
    if set(x["split"] for x in records)!={"train","val","test"}: raise ValueError("representative manifest needs all patient splits")
    manifest={"schema_version":"semantic_phase3_v1","identity_classes":CLASSES,"tissue_classes":["unavailable_0","unavailable_1"],"provenance":{"feature_kind":"he_only_xcell_reg64","sampling":"complete_detected_inventory","nuclei_class_source":"unavailable","detector_version":"cellpose_cpsam_from_expert_cache","xcell_checkpoint_sha256":meta["input_probe_metadata_sha256"],"phase2_checkpoint_sha256":sha256_file(a.phase2_checkpoint),"ontology_version":"crc_panel2_fine5_expert_reviewed_v1","coordinate_version":"registered_he_dapi_mutual_nearest_le4px","label_source":"doctor_reviewed_mif_transferred_to_he","registration_qc":"cell_correspondence_verified","area_note":"physical MPP unavailable; positive placeholder is forbidden from composition/fusion supervision"},"samples":records}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); (out/"metadata.json").write_text(json.dumps({"status":"complete","records":len(records),"per_case":seen,"source_cache":str(a.cache_dir),"manifest_sha256":sha256_file(out/"manifest.json"),"phase2_checkpoint":str(a.phase2_checkpoint),"identity_only":True},indent=2))
    print(json.dumps({"status":"complete","output":str(out),"records":len(records)}),flush=True)
if __name__=="__main__": main()
