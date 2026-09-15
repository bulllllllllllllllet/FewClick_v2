#!/usr/bin/python3
"""Embed blinded or assisted CRC H&E/mIF review evidence into one Excel workbook."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue

BLIND_DECISIONS = ["CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage",
             "IDENTITY_UNCERTAIN", "REGISTRATION_ERROR", "IMAGE_QUALITY_BAD"]
ASSISTED_DECISIONS = ["CONFIRM_AUTO", "CHANGE_LABEL", "IDENTITY_UNCERTAIN",
                      "REGISTRATION_ERROR", "IMAGE_QUALITY_BAD"]
CELL_CLASSES = ["CD4_T", "CD8_T", "Treg", "B_cell", "CD163_Macrophage"]
CONFIDENCE = ["high", "medium", "low"]
# LibreOffice exposes these UNO enum properties as integers through its bundled Python bridge.
VALIDATION_LIST = 6
VALIDATION_ALERT_STOP = 0
VERTICAL_CENTER = 1


def prop(name, value):
    item = PropertyValue()
    item.Name, item.Value = name, value
    return item


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect_office(profile):
    port = free_port()
    command = ["/usr/bin/libreoffice", f"-env:UserInstallation={uno.systemPathToFileUrl(profile)}",
               "--headless", "--nologo", "--nodefault", "--nofirststartwizard", "--norestore",
               f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ServiceManager"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    deadline, context = time.time() + 20, None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("LibreOffice exited before UNO connection")
        try:
            context = resolver.resolve(f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext")
            break
        except Exception:
            time.sleep(.2)
    if context is None:
        process.terminate()
        raise TimeoutError("LibreOffice UNO connection timeout")
    desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
    return process, desktop


def set_list_validation(cell_range, values, message):
    validation = cell_range.Validation
    validation.Type = VALIDATION_LIST
    validation.Formula1 = '"' + ";".join(values) + '"'
    validation.IgnoreBlankCells = True
    validation.ShowList = 1
    validation.ShowErrorMessage = True
    validation.ErrorTitle = "输入不在允许范围"
    validation.ErrorMessage = message
    validation.ErrorAlertStyle = VALIDATION_ALERT_STOP
    cell_range.Validation = validation


def build_workbook(desktop, source, output, mode, limit=0):
    input_name = "review_assignment_private.csv" if mode == "assisted" else "review_results_template.csv"
    with (source / input_name).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if limit:
        rows = rows[:limit]
    if not rows or len({row["review_id"] for row in rows}) != len(rows):
        raise ValueError("review rows must be nonempty and unique")
    hidden = (prop("Hidden", True),)
    document = desktop.loadComponentFromURL("private:factory/scalc", "_blank", 0, hidden)
    sheets = document.Sheets
    instructions = sheets.getByIndex(0); instructions.Name = "说明"
    sheets.insertNewByName("复核", 1); review = sheets.getByName("复核")

    instruction_lines = [
        "CRC H&E/mIF 细胞身份辅助复核" if mode == "assisted" else "CRC H&E/mIF 细胞身份盲法复核",
        "本文件已嵌入全部证据图，不依赖HTML、外部图片或网络。请先另存为包含审核人ID的文件名。",
        "审核人ID（请在右侧黄色单元格填写）：",
        "红十字=H&E中心，青色=H&E边界，黄色=DAPI目标边界。",
        ("核验原自动类别：确认则选CONFIRM_AUTO；不同则选CHANGE_LABEL并填写修正类别。"
         if mode == "assisted" else "人工判定只能从下拉列表选择：五类、身份不确定、配准错误、图像质量差。"),
        "置信度选择high/medium/low，表示你对本次审核结论的把握，不是模型概率。",
        "审核完成后直接保存xlsx。两名审核者分别保存独立文件，不互看结果。",
        ("该工作簿显示原自动标签以便核验；人工选择也不会自动修改源数据或提升为verified标签。"
         if mode == "assisted" else "该工作簿不包含原自动标签；人工选择也不会自动修改源数据或提升为verified标签。"),
    ]
    for index, value in enumerate(instruction_lines):
        instructions.getCellByPosition(0, index).String = value
    instructions.getCellByPosition(1, 2).CellBackColor = 0xFFF2CC
    instructions.getCellByPosition(1, 2).String = ""
    instructions.getCellRangeByName("A1:B1").merge(True)
    instructions.getCellByPosition(0, 0).CharWeight = 150
    instructions.getCellByPosition(0, 0).CharHeight = 18
    instructions.Columns.getByIndex(0).Width = 24000
    instructions.Columns.getByIndex(1).Width = 7000
    for index in range(len(instruction_lines)):
        instructions.Rows.getByIndex(index).Height = 800

    if mode == "assisted":
        headers = ["review_id", "原自动类别", "证据图（H&E、DAPI、CD4、CD8、CD20、CD163、FOXP3、SULF1）",
                   "审核结论", "修正类别（仅CHANGE_LABEL时填写）", "置信度", "备注"]
        image_column, decision_column, correction_column, confidence_column, note_column = 2, 3, 4, 5, 6
        widths = [2400, 4200, 20000, 4200, 5500, 2800, 9000]
    else:
        headers = ["review_id", "证据图（H&E、DAPI、CD4、CD8、CD20、CD163、FOXP3、SULF1）",
                   "人工判定", "置信度", "备注"]
        image_column, decision_column, correction_column, confidence_column, note_column = 1, 2, None, 3, 4
        widths = [2400, 20000, 5000, 2800, 9000]
    for column, value in enumerate(headers):
        cell = review.getCellByPosition(column, 0)
        cell.String = value; cell.CharWeight = 150; cell.CellBackColor = 0xD9EAF7
    for column, width in enumerate(widths):
        review.Columns.getByIndex(column).Width = width
    review.Rows.getByIndex(0).Height = 900
    for offset, row in enumerate(rows, 1):
        review.Rows.getByIndex(offset).Height = 3200
        review.getCellByPosition(0, offset).String = row["review_id"]
        if mode == "assisted":
            review.getCellByPosition(1, offset).String = row["class"]
        review.getCellByPosition(decision_column, offset).CellBackColor = 0xFFF2CC
        if correction_column is not None:
            review.getCellByPosition(correction_column, offset).CellBackColor = 0xFFF2CC
        review.getCellByPosition(confidence_column, offset).CellBackColor = 0xFFF2CC
        image_path = source / "images" / f"{row['review_id']}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        anchor = review.getCellByPosition(image_column, offset)
        shape = document.createInstance("com.sun.star.drawing.GraphicObjectShape")
        shape.GraphicURL = uno.systemPathToFileUrl(str(image_path.resolve()))
        shape.Size = Size(19500, 2888)
        review.DrawPage.add(shape)
        shape.Anchor = anchor
        position = anchor.Position
        shape.Position = Point(position.X + 150, position.Y + 140)
    end = len(rows) + 1
    decision_letter = chr(ord("A") + decision_column)
    confidence_letter = chr(ord("A") + confidence_column)
    set_list_validation(review.getCellRangeByName(f"{decision_letter}2:{decision_letter}{end}"),
                        ASSISTED_DECISIONS if mode == "assisted" else BLIND_DECISIONS, "请从预设结论中选择。")
    if correction_column is not None:
        correction_letter = chr(ord("A") + correction_column)
        set_list_validation(review.getCellRangeByName(f"{correction_letter}2:{correction_letter}{end}"), CELL_CLASSES,
                            "仅在CHANGE_LABEL时选择修正类别。")
    set_list_validation(review.getCellRangeByName(f"{confidence_letter}2:{confidence_letter}{end}"), CONFIDENCE,
                        "请选择high、medium或low。")
    review.getCellRangeByName(f"A1:{chr(ord('A') + note_column)}{end}").VertJustify = VERTICAL_CENTER
    document.CurrentController.setActiveSheet(review)
    document.CurrentController.freezeAtPosition(0, 1)
    target = uno.systemPathToFileUrl(str(output.resolve()))
    document.storeAsURL(target, (prop("FilterName", "Calc MS Excel 2007 XML"), prop("Overwrite", False)))
    document.close(True)
    return rows


def reopen_audit(desktop, workbook, expected_rows, mode):
    document = desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(workbook.resolve())), "_blank", 0,
                                            (prop("Hidden", True), prop("ReadOnly", True)))
    try:
        assert document.Sheets.hasByName("说明") and document.Sheets.hasByName("复核")
        sheet = document.Sheets.getByName("复核")
        first = sheet.getCellByPosition(0, 1).String
        last = sheet.getCellByPosition(0, expected_rows).String
        drawings = sheet.DrawPage.Count
        decision_column, confidence_column = (3, 5) if mode == "assisted" else (2, 3)
        decision_letter = chr(ord("A") + decision_column)
        confidence_letter = chr(ord("A") + confidence_column)
        decision_type = str(sheet.getCellRangeByName(f"{decision_letter}2:{decision_letter}{expected_rows+1}").Validation.Type)
        confidence_type = str(sheet.getCellRangeByName(f"{confidence_letter}2:{confidence_letter}{expected_rows+1}").Validation.Type)
        correction_type = None
        if mode == "assisted":
            correction_type = str(sheet.getCellRangeByName(f"E2:E{expected_rows+1}").Validation.Type)
        if first != "R0001" or last != f"R{expected_rows:04d}" or drawings != expected_rows:
            raise AssertionError((first, last, drawings, expected_rows))
        if "'LIST'" not in decision_type or "'LIST'" not in confidence_type or (correction_type is not None and "'LIST'" not in correction_type):
            raise AssertionError((decision_type, correction_type, confidence_type))
        return {"sheets": list(document.Sheets.ElementNames), "first_id": first, "last_id": last,
                "embedded_drawings": drawings, "decision_validation": "LIST", "confidence_validation": "LIST",
                "correction_validation": "LIST" if mode == "assisted" else None}
    finally:
        document.close(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", choices=("blinded", "assisted"), default="blinded")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}_\d{6}", args.timestamp) or args.limit < 0:
        raise ValueError("valid timestamp and nonnegative limit required")
    prefix = "crc_manual_review_assisted_excel" if args.mode == "assisted" else "crc_manual_review_excel"
    root = Path("/nfs-medical3/zyh/fewclick_v2/semantic_phase3") / f"{prefix}_{args.timestamp}"
    root.mkdir(exist_ok=False)
    workbook = root / f"CRC_manual_review_{args.mode}.xlsx"
    profile = tempfile.mkdtemp(prefix="crc_review_lo_")
    process = None
    try:
        process, desktop = connect_office(profile)
        rows = build_workbook(desktop, args.source.resolve(), workbook, args.mode, args.limit)
        reopened = reopen_audit(desktop, workbook, len(rows), args.mode)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        shutil.rmtree(profile)
    with zipfile.ZipFile(workbook) as archive:
        names = archive.namelist()
        media = [name for name in names if name.startswith("xl/media/")]
        text = b"".join(archive.read(name) for name in names if name.endswith(".xml"))
    for forbidden in [b"2017-23800D", b"2019-00538F", b"2022-39941E", b"geometry_risk"]:
        if forbidden in text:
            raise AssertionError(f"private source field leaked into blinded workbook: {forbidden!r}")
    if len(media) != len(rows):
        raise AssertionError((len(media), len(rows)))
    result = {"status": "passed", "rows": len(rows), "embedded_media": len(media), **reopened,
              "workbook": str(workbook), "bytes": workbook.stat().st_size,
              "sha256": hashlib.sha256(workbook.read_bytes()).hexdigest(),
              "mode": args.mode,
              "label_visibility": "per-row auto label visible" if args.mode == "assisted" else "no per-row auto label stored",
              "private_fields_excluded": "case, risk stratum and marker value",
              "scope": "manual review input; reviewer output does not automatically become verified truth"}
    (root / "workbook_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
