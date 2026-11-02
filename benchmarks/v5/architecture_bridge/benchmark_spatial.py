"""Fixed GPU throughput probe; no development metric or model selection."""
import argparse
import json
import os
from pathlib import Path
import time
import pyvips
import torch
from .spatial_controls import SpatialModel,SpatialData
from .restored_loss import episode_loss
from benchmarks.v5.encoder_research.continuous_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    proof=json.loads((a.protocol/'acceptance.json').read_text());assert sha(a.protocol/'training_consumption.jsonl')==proof['consumption_sha256']
    data=SpatialData(a.cache,'S11');model=SpatialModel('S11');model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    plans=list(map(json.loads,(a.protocol/'training_consumption.jsonl').read_text().splitlines()))[:80];times=[]
    for step in range(20):
        torch.cuda.synchronize();started=time.perf_counter();opt.zero_grad(set_to_none=True)
        for task in plans[step*4:step*4+4]:
            assert data.rows[task['query_id']]['split']==data.rows[task['support_id']]['split']=='train'
            inputs=data.model_inputs(task,'xcell','cuda:0');loss=episode_loss(model(**inputs),data,task,inputs['prompts']);assert torch.isfinite(loss);(loss/4).backward()
        opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-started)
        print(json.dumps(dict(stage='benchmark_train',updates=step+1,total_updates=20,worker_pid=os.getpid())),flush=True)
    model.eval();evaluation=list(map(json.loads,(a.protocol/'evaluation.jsonl').read_text().splitlines()));selected=[]
    for scope,dataset in [('local','crc'),('local','tiger'),('remote','crc'),('remote','tiger')]:
        selected += [x for x in evaluation if x['scope']==scope and x['budget']=='3+2' and x['plan']['dataset']==dataset][:25]
    assert len(selected)==100;inference=[]
    with torch.no_grad():
        for i,item in enumerate(selected):
            torch.cuda.synchronize();started=time.perf_counter()
            output=model(**data.model_inputs(item['plan'],'xcell','cuda:0'))['query']['pixel_logits'];assert torch.isfinite(output).all()
            torch.cuda.synchronize();inference.append(time.perf_counter()-started)
            if (i+1)%10==0:print(json.dumps(dict(stage='benchmark_inference',rows=i+1,total_rows=100)),flush=True)
    (a.output/'report.json').write_text(json.dumps(dict(status='accepted_S11_GPU_throughput_probe',updates=20,tasks=80,inference_tasks=100,train_seconds=times,
        inference_seconds=inference,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),source_sha256=sha(__file__),protocol_sha256=sha(a.protocol/'acceptance.json'),
        qualification='timings include data loading and image encoding; exclude checkpoint/model loading, BF1 scoring and fixed train monitors; no validation GT read; not WSI/online latency or quality result'),indent=2)+'\n')


if __name__=='__main__':main()
