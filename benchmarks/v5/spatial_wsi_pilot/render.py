"""Repair display of absent-target Dice using saved predictions; no new inference."""
import argparse,json,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root;rows=json.load(open(r/'rows.json'));inventory=json.load(open(r/'inventory.json'))
 html='<meta charset="utf-8"><h1>WSI索引验证空间块</h1><p>全片仅缩略图展示；预测/计分仅限已审计验证空间块，并非连续全片扫描。每片固定一组支持点击，查询不加人工点。空目标不报告Dice，报告FPA。</p>'
 for i,item in enumerate(inventory):html+=f'<h2>{item["slide"]}</h2><p>目标ID {item["target"]}；查询位置 {len(item["query_ids"])}</p><img width="640" src="slide_{i}.png">'
 html+='<h2>支持区（绿正点/紫负点）｜查询原图｜GT黄｜预测红</h2>'
 for j,row in enumerate(rows):
  value=row['metrics']['dice'];label=f'Dice {value:.4f}' if value is not None else f'无目标，FPA {row["metrics"]["false_positive_fraction"]:.4f}'
  assert (r/f'prediction_{j}.png').exists();html+=f'<p>{row["plan"]["query_id"]}，{label}</p><img width="1200" src="prediction_{j}.png">'
 (r/'index.html').write_text(html);(r/'render_acceptance.json').write_text(json.dumps(dict(status='accepted_display_repair',rows=len(rows),absent=sum(x['metrics']['dice'] is None for x in rows),html_sha256=hashlib.sha256((r/'index.html').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inference_repeated=False))+'\n')
if __name__=='__main__':main()
