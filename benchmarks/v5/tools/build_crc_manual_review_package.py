#!/usr/bin/env python3
"""Build a blinded, stratified CRC H&E/mIF cell review package."""
from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries
import tifffile
import zarr

CLASSES = ["CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage"]
MARKERS = ["DAPI", "CD4", "CD8", "CD20", "CD163", "FOXP3", "SULF1"]
DISPLAY_MAX = {"DAPI": 6500, "CD4": 2000, "CD8": 1200, "CD20": 1200,
               "CD163": 1264, "FOXP3": 3500, "SULF1": 3000}
CHANNELS = {"DAPI": "DAPI", "Opal 480": "CD163", "Opal 520": "FOXP3",
            "Opal 570": "CD20", "Opal 620": "CD8", "Opal 690": "CD4",
            "Opal 780": "SULF1", "Sample AF": "AF"}


def add_stratum(frame):
    frame = frame.copy()
    frame["review_stratum"] = "label_rule_risk"
    frame.loc[~frame.geometry_review_pass, "review_stratum"] = "geometry_risk"
    frame.loc[frame.geometry_review_pass & frame.stable_single_marker,
              "review_stratum"] = "joint_diagnostic_pass"
    return frame


def stratified_sample(frame, per_case_class=24, seed=20260915):
    """Census Treg; fixed quota for every other case/class, preserving risk strata."""
    frame = add_stratum(frame)
    rng = np.random.default_rng(seed)
    selected = []
    strata = ["joint_diagnostic_pass", "geometry_risk", "label_rule_risk"]
    for (case, label), group in frame.groupby(["case", "class"], sort=True):
        if label == "Treg":
            take = group.copy()
        else:
            target = min(per_case_class, len(group))
            indices = []
            base = target // len(strata)
            for stratum in strata:
                pool = group[group.review_stratum == stratum].index.to_numpy()
                count = min(base, len(pool))
                if count:
                    indices.extend(rng.choice(pool, count, replace=False).tolist())
            remaining = group.drop(index=indices)
            if len(indices) < target:
                indices.extend(rng.choice(remaining.index.to_numpy(), target-len(indices), replace=False).tolist())
            take = group.loc[indices].copy()
        selected.append(take)
    result = pd.concat(selected).copy()
    population = frame.groupby(["case", "class", "review_stratum"]).size()
    sampled = result.groupby(["case", "class", "review_stratum"]).size()
    keys = list(zip(result.case, result["class"], result.review_stratum))
    result["stratum_population"] = [int(population[key]) for key in keys]
    result["stratum_sampled"] = [int(sampled[key]) for key in keys]
    result["selection_probability_within_candidate_pool"] = result.stratum_sampled / result.stratum_population
    result["inverse_probability_weight"] = 1 / result.selection_probability_within_candidate_pool
    result = result.iloc[rng.permutation(len(result))].reset_index(drop=True)
    result.insert(0, "review_id", [f"R{i:04d}" for i in range(1, len(result)+1)])
    return result


def channel_map(tif):
    names = []
    for page in tif.pages:
        try:
            root = ET.fromstring(page.description or "")
        except ET.ParseError:
            continue
        if root.findtext("ImageType") == "FullResolution":
            names.append(root.findtext("Name"))
    mapping = {CHANNELS[name]: i for i, name in enumerate(names)}
    if len(mapping) != 8:
        raise ValueError(f"unexpected qptiff channels: {mapping}")
    return mapping


def crop(array, cx, cy, radius=32):
    if array.ndim == 2:
        canvas = np.zeros((radius*2+1, radius*2+1), dtype=array.dtype)
    else:
        canvas = np.zeros((radius*2+1, radius*2+1, array.shape[2]), dtype=array.dtype)
    x0, x1 = max(0, cx-radius), min(array.shape[1], cx+radius+1)
    y0, y1 = max(0, cy-radius), min(array.shape[0], cy+radius+1)
    dx, dy = x0-(cx-radius), y0-(cy-radius)
    canvas[dy:dy+y1-y0, dx:dx+x1-x0] = array[y0:y1, x0:x1]
    return canvas


def overlay_boundaries(rgb, he_boundary, dapi_boundary):
    rgb = rgb.copy()
    rgb[he_boundary] = [0, 255, 255]
    rgb[dapi_boundary] = [255, 230, 0]
    center = rgb.shape[0] // 2
    rgb[max(0, center-3):center+4, center, :] = [255, 0, 0]
    rgb[center, max(0, center-3):center+4, :] = [255, 0, 0]
    return rgb


def panel_image(he, stack, mapping, he_mask, dapi, row):
    cx, cy = int(round(row.x)), int(round(row.y))
    he_crop = crop(he, cx, cy)
    he_boundary = find_boundaries(crop(he_mask == int(row.he_id), cx, cy), mode="inner")
    dapi_boundary = find_boundaries(crop(dapi == int(row.dapi_id), cx, cy), mode="inner")
    panels = [overlay_boundaries(he_crop, he_boundary, dapi_boundary)]
    for marker in MARKERS:
        gray = crop(stack[mapping[marker]], cx, cy).astype(np.float32)
        gray = (np.clip(gray / DISPLAY_MAX[marker], 0, 1) * 255).astype(np.uint8)
        panels.append(overlay_boundaries(np.repeat(gray[:, :, None], 3, axis=2), he_boundary, dapi_boundary))
    scale, top = 2, 24
    width = len(panels) * panels[0].shape[1] * scale
    canvas = Image.new("RGB", (width, panels[0].shape[0]*scale + top), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (array, title) in enumerate(zip(panels, ["H&E", *MARKERS])):
        image = Image.fromarray(array).resize((array.shape[1]*scale, array.shape[0]*scale), Image.Resampling.NEAREST)
        x = index * image.width
        canvas.paste(image, (x, top))
        draw.text((x+4, 6), title, fill="black", font=font)
    return canvas


def html_page(items):
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>CRC盲法细胞复核</title>
<style>body{font-family:sans-serif;max-width:1150px;margin:20px auto;background:#f5f5f5}#card{background:white;padding:18px;border-radius:10px}img{width:100%;image-rendering:auto}.choices{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}button,label,select,input,textarea{font-size:15px}label.choice{border:1px solid #bbb;padding:8px;background:#fafafa}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.hidden{display:none}.warn{color:#9b2c2c}.small{font-size:13px;color:#555}textarea{width:100%;height:50px}</style></head><body>
<h2>CRC H&E/mIF 细胞身份盲法复核</h2><p>红十字=H&E检测中心；青色=H&E实例边界；黄色=目标DAPI实例边界。marker采用固定显示范围。先独立选择，必要时再揭示自动标签。</p>
<div class='bar'>审核人ID <input id='reviewer' placeholder='必填，例如 reviewer_A'> <span id='progress'></span><button onclick='downloadCSV()'>下载CSV</button><button onclick='clearAll()'>清空本浏览器结果</button></div><hr>
<div id='card'><h3 id='rid'></h3><img id='evidence'><div id='source' class='hidden warn'></div><button onclick='toggleSource()'>显示/隐藏原自动标签与数值</button>
<div class='choices' id='choices'></div><div>置信度 <select id='confidence'><option value=''>请选择</option><option>high</option><option>medium</option><option>low</option></select></div><p>备注</p><textarea id='note'></textarea>
<div class='bar'><button onclick='move(-1)'>上一项</button><button onclick='saveAndNext()'>保存并下一项</button><button onclick='move(1)'>跳过/下一项</button><input id='jump' type='number' min='1'><button onclick='jumpTo()'>跳转</button></div></div>
<p class='small'>结果只保存在当前浏览器localStorage，必须定期下载CSV。不同审核人应使用独立浏览器配置或下载后清空。此界面不会修改源数据，也不会自动生成verified训练标签。</p>
<script>const items=""" + data + """; const decisions=['CD4_T','CD8_T','Treg','B_cell','CD163_Macrophage','IDENTITY_UNCERTAIN','REGISTRATION_ERROR','IMAGE_QUALITY_BAD'];
const rid=document.getElementById('rid'),evidence=document.getElementById('evidence'),source=document.getElementById('source'),choices=document.getElementById('choices'),confidence=document.getElementById('confidence'),note=document.getElementById('note'),reviewer=document.getElementById('reviewer'),progress=document.getElementById('progress'),jump=document.getElementById('jump');
let idx=Number(localStorage.getItem('crc_review_index')||0), answers=JSON.parse(localStorage.getItem('crc_review_answers')||'{}');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){idx=Math.max(0,Math.min(items.length-1,idx));const x=items[idx],a=answers[x.review_id]||{};rid.textContent=`${x.review_id} (${idx+1}/${items.length})`;evidence.src=x.image;source.classList.add('hidden');source.innerHTML=`原自动标签：<b>${esc(x.auto_label)}</b>；病例：${esc(x.case)}；层：${esc(x.stratum)}<br>${esc(x.intensities)}`;choices.innerHTML=decisions.map((d,i)=>`<label class='choice'><input type='radio' name='decision' value='${d}' ${a.decision===d?'checked':''}> ${i+1}. ${d}</label>`).join('');confidence.value=a.confidence||'';note.value=a.note||'';reviewer.value=localStorage.getItem('crc_reviewer')||'';progress.textContent=`已完成 ${Object.keys(answers).length}/${items.length}`;localStorage.setItem('crc_review_index',idx)}
function save(){const d=document.querySelector('input[name=decision]:checked');if(!d||!confidence.value||!reviewer.value.trim()){alert('请填写审核人ID、判定和置信度');return false}localStorage.setItem('crc_reviewer',reviewer.value.trim());answers[items[idx].review_id]={decision:d.value,confidence:confidence.value,note:note.value,reviewer:reviewer.value.trim(),reviewed_at:new Date().toISOString()};localStorage.setItem('crc_review_answers',JSON.stringify(answers));return true}
function saveAndNext(){if(save())move(1)} function move(n){idx+=n;render()} function jumpTo(){idx=Number(jump.value)-1;render()} function toggleSource(){source.classList.toggle('hidden')}
function csvCell(v){return '"'+String(v??'').replaceAll('"','""')+'"'} function downloadCSV(){const h=['review_id','reviewer','decision','confidence','note','reviewed_at'];let rows=[h.join(',')];for(const x of items){const a=answers[x.review_id]||{};rows.push(h.map(k=>csvCell(k==='review_id'?x.review_id:a[k])).join(','))}const blob=new Blob([rows.join('\\n')],{type:'text/csv;charset=utf-8'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=`crc_review_${localStorage.getItem('crc_reviewer')||'unknown'}.csv`;a.click();URL.revokeObjectURL(u)}
function clearAll(){if(confirm('确认清空当前浏览器的全部审核结果？请先下载CSV。')){localStorage.removeItem('crc_review_answers');localStorage.removeItem('crc_review_index');answers={};idx=0;render()}}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;const n=Number(e.key);if(n>=1&&n<=8){document.querySelectorAll('input[name=decision]')[n-1].checked=true}if(e.key==='Enter')saveAndNext()});render();</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--per-case-class", type=int, default=24)
    parser.add_argument("--cases", nargs="+", choices=("2017-23800D", "2019-00538F", "2022-39941E"),
                        default=["2017-23800D", "2019-00538F", "2022-39941E"])
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp):
        raise ValueError("timestamp required")
    out = Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3") / f"crc_manual_review_{args.timestamp}"
    out.mkdir(exist_ok=False); image_dir = out / "images"; image_dir.mkdir()
    candidates = pd.read_csv(args.audit_dir / "candidate_audit.csv")
    candidates = candidates[candidates.case.isin(args.cases)].copy()
    selected = stratified_sample(candidates, args.per_case_class)
    probe = json.loads((args.probe_dir / "metadata.json").read_text())
    records = {(r["case_id"], r["patch_name"]): r for r in probe["records"]}
    reg_root = Path(probe["registration_root"])
    rendered = 0
    for case, case_frame in selected.groupby("case", sort=True):
        qpath = Path("/nfs-medical6/data/SRRSH/crc/panel_2/mIF") / case / "Scan1" / f"{case}_Scan1.er.qptiff"
        with tifffile.TiffFile(qpath) as tif:
            mapping = channel_map(tif); store = tif.series[0].aszarr(); source = zarr.open(store, mode="r")["0"]
            for patch, frame in case_frame.groupby("patch", sort=True):
                record = records[(case, patch)]
                he = np.asarray(Image.open(record["he_path"]).convert("RGB"))
                dapi = np.load(reg_root / case / "dapi_masks" / f"{patch}.npy", allow_pickle=False)
                match = re.fullmatch(r"x(\d+)_y(\d+)", patch); x0, y0 = map(int, match.groups())
                stack = np.asarray(source[:, y0:y0+dapi.shape[0], x0:x0+dapi.shape[1]])
                with np.load(args.probe_dir / record["path"], allow_pickle=False) as cache:
                    he_mask = cache["he_mask"]
                for row in frame.itertuples():
                    path = image_dir / f"{row.review_id}.jpg"
                    panel_image(he, stack, mapping, he_mask, dapi, row).save(path, quality=92)
                    rendered += 1
            store.close()
    selected["image"] = selected.review_id.map(lambda x: f"images/{x}.jpg")
    selected.to_csv(out / "review_assignment_private.csv", index=False)
    public = selected[["review_id", "image"]].copy()
    for field in ["reviewer", "decision", "confidence", "note", "reviewed_at"]:
        public[field] = ""
    public.to_csv(out / "review_results_template.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    html_items = []
    for _, row in selected.iterrows():
        intensity = ", ".join(f"{m}={row[m+'_intensity']:.1f}" for m in MARKERS[1:])
        html_items.append({"review_id": row.review_id, "image": row.image, "auto_label": row["class"],
                           "case": row.case, "stratum": row.review_stratum, "intensities": intensity})
    (out / "review.html").write_text(html_page(html_items), encoding="utf-8")
    (out / "README.md").write_text("""# CRC盲法细胞身份复核\n\n1. 使用Chrome/Edge打开`review.html`，填写唯一审核人ID。\n2. 先看八联图独立选择五类之一，或选择身份不确定、配准错误、图像质量差；同时选择置信度。\n3. 只有确有需要时才点击揭示原自动标签，避免默认答案造成锚定偏差。数字键1–8选择，Enter保存并进入下一项。\n4. 浏览器localStorage只在本机保存，审核中和结束时均点击“下载CSV”。不同审核人应独立完成，不共享前一人的结果。\n5. `review_assignment_private.csv`含自动标签和风险层，仅供数据管理员，不应交给盲法审核者。\n6. 审核结果不会自动修改源标签；两名审核人结果需检查一致性，不一致项再仲裁。\n\n边界颜色：红十字为H&E检测中心，青色为H&E实例边界，黄色为目标DAPI实例边界。marker使用固定显示范围；图像是证据视图，不是模型预测。\n""", encoding="utf-8")
    counts = selected.groupby(["case", "class", "review_stratum"]).size().rename("n").reset_index()
    counts.to_csv(out / "selection_summary.csv", index=False)
    audit = {"status": "passed", "samples": len(selected), "rendered": rendered,
             "unique_review_ids": selected.review_id.nunique(), "unique_source_cells": selected[["case", "patch", "he_id"]].drop_duplicates().shape[0],
             "by_class": selected["class"].value_counts().to_dict(), "by_case": selected.case.value_counts().to_dict(),
             "requested_cases": args.cases, "per_case_class_non_treg": args.per_case_class,
             "treg_is_census_of_candidate_pool": int((selected["class"] == "Treg").sum()) == int((candidates["class"] == "Treg").sum()),
             "blinding": "auto label hidden by default but revealable", "source_candidate_pool": str(args.audit_dir),
             "scope": "review package only; reviewer decisions are not automatically verified training labels"}
    assert rendered == len(selected) == selected.review_id.nunique() == audit["unique_source_cells"]
    assert all((image_dir / f"{rid}.jpg").exists() for rid in selected.review_id)
    (out / "package_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
