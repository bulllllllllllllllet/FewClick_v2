"""Separate weight identity and source claims from actual exposure evidence."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmarks.v5.fair_comparison.audit import digest,write


def run(catalog,output):
    output.mkdir(parents=True,exist_ok=False)
    weights={'xcell':Path('/home/zhaoyh/FewClick/module/checkpoint/he_model_best.pth'),
             'ctranspath':Path('/home/zhaoyh/FewClick/module/checkpoint/ctranspath.pth'),
             'instanseg':Path('/nfs-medical3/zyh/fewclick_v2/multidomain_data/detector_speed_1020_182710/instanseg_model/instanseg.pt')}
    records={}
    for name,path in weights.items():
        records[name]=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)
    xcell=torch.load(weights['xcell'],map_location='cpu',weights_only=True)
    records['xcell'].update(payload_type=type(xcell).__name__,entries=len(xcell),
        all_tensors=all(isinstance(x,torch.Tensor) for x in xcell.values()),
        non_tensor_metadata_keys=[k for k,v in xcell.items() if not isinstance(v,torch.Tensor)],
        checkpoint_linked_training_manifest=None,
        training_source='internal dataset, not current CRC/TIGER',
        training_source_evidence='explicit user clarification2026-10-25',
        current_CRC_TIGER_used_for_XCell_training=False,
        overlap_status='user-confirmed distinct XCell training cohort; upstream CTransPath prior is separate')
    old=Path('/home/zhaoyh/FewClick');claims=old/'benchmarks/gdph_v2/experiment.py'
    if '"external_to_xcellformer_training": True' not in claims.read_text():raise ValueError('historical source claim changed')
    records['historical_GDPH_external_claim']=dict(path=str(claims),sha256=digest(claims),
        evidence_kind='hardcoded configuration assertion, not an audit result',proves_no_overlap=False)
    missing=[old/'experiments/crc012_holdout3/20260515_160711',Path('/home/zhaoyh/CellAtlas/experiments/crc012_holdout3/20260515_160711'),
             Path('/nfs-medical3/zyh/cellatlas_gdph_benchmark_v2/FINAL_REPORT.md')]
    records['referenced_history_locations']=[dict(path=str(p),exists=p.exists()) for p in missing]
    training=old/'new_training_stream/multidata_aligner_trainer.py'
    records['available_XCell_training_implementation']=dict(path=str(training),sha256=digest(training),
        losses='row-aligned HE/mIF MSE plus contrastive',artifact_format='he_model.state_dict only',
        checkpoint_link_to_this_execution='not established',
        heldout_policy='code can select checkpoints by heldout-task ValMSE; a holdout name alone does not prove blind testing')
    table=pd.read_parquet(catalog/'tiles.parquet');tiger=table[table.dataset=='tiger']
    records['ctranspath'].update(author_source='https://github.com/Xiyue-Wang/TransPath',
        reported_image_pretraining='TCGA and PAIP',local_README_sha256=digest(Path('module/TransPath/README.md')),
        current_TIGER_rows=len(tiger),current_TIGER_TCGA_spaces=sorted(set(k for k in tiger.space_key if 'TCGA-' in k)),
        exact_pretraining_slide_manifest=None,overlap_status='shared cohort family possible; exact slide/region overlap not verified')
    records['wsi_sam']=dict(author_source='https://arxiv.org/html/2403.09257v2#S3',
        paper_reported_training='CATCH canine cutaneous tumors; 245 training and35 validation WSIs',
        current_released_checkpoint_training_manifest=None,
        conclusion='different reported cohort from current human CRC/TIGER; do not infer an artifact-level no-overlap proof')
    records['cell_inference_contract']=dict(tissue_GT_used=False,nuclear_identity='all unknown, feature column3=1; remaining class onehot zeros',
        source=['benchmarks/v5/multidomain_data/prepare_cells.py','benchmarks/v5/paper_multiscale/prepare.py'],
        limitation='inference GT exclusion does not establish pretraining exposure exclusion')
    report=dict(status='complete',catalog_sha256=digest(catalog/'tiles.parquet'),records=records,
        allowed=['image-only control excludes cell appearance prior','morphology/count control excludes XCell64 input information',
                 'report public/generic priors and incremental supervision separately'],
        disallowed=['entire feature chain proven unseen at every image-pretraining stage','XCell gain isolates molecular supervision',
                    'old GDPH config assertion substitutes a training manifest'],
        unresolved=['archival checkpoint-linked XCell training/selection manifest (source cohort already user-confirmed)','exact CTransPath slide exposure',
                    'released WSI-SAM artifact-linked training manifest'])
    write(output/'audit.json',report)
    (output/'report.md').write_text('''# 预训练证据分级审计

用户确认：XCell使用内部数据集训练，当前CRC/TIGER未参与该训练。按这一来源说明更新账本。当前权重只有71个tensor，未携带训练清单；旧GDPH external字段是源码断言。文件级训练/选模清单尚未归档，只记为可复现档案缺口，不把它当作与用户说明相反的证据。

CTransPath作者README报告TCGA/PAIP预训练；当前TIGER存在TCGA来源。缺少逐片清单时只能记可能同源，不能写成已证实重叠，也不能写严格未见所有图像。[作者来源](https://github.com/Xiyue-Wang/TransPath)

WSI-SAM论文报告CATCH犬皮肤肿瘤训练，未报告在当前人CRC/TIGER上训练；发布权重没有提供可直接绑定的完整训练清单。两项事实分开记录。[论文训练数据](https://arxiv.org/html/2403.09257v2#S3)

当前细胞提取路径只用图像，核类型全unknown；这证明推理输入合同，不证明预训练隔离。保留image-only和morphology/count重训控制，不能将XCell与形态差异直接解释为分子监督因果贡献。未知项不会被自动标为无泄漏。
''')
    print(json.dumps(dict(status='complete',output=str(output),unresolved=report['unresolved'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.catalog,a.output)
