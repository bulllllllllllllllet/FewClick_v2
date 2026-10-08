#!/usr/bin/env python3
"""Download a user-authorized CytoFormer checkpoint with size and digest checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

BASE=Path('/nfs-medical3/zyh/fewclick_v2/semantic_phase3')
REPO='zhihuanglab/CytoFormer'
FILE='checkpoint.pth'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(BASE):
        p.error('output must be inside FewClick artifact root')
    if not get_token():
        raise ValueError('no user-authorized Hugging Face token configured')
    meta=get_hf_file_metadata(hf_hub_url(REPO,FILE),timeout=30)
    if not meta.size or not re.fullmatch(r'[0-9a-f]{64}',meta.etag or ''):
        raise ValueError('checkpoint metadata has no verifiable size/SHA256')
    a.output.mkdir(parents=True,exist_ok=False)
    partial=a.output/f'{FILE}.partial'
    h=hashlib.sha256();total=0;next_report=128<<20
    with requests.get(meta.location,stream=True,timeout=(15,120)) as response:
        response.raise_for_status()
        if response.status_code!=200 or int(response.headers.get('Content-Length','-1'))!=meta.size:
            raise ValueError('CDN response size/status mismatch')
        with partial.open('xb') as output:
            for chunk in response.iter_content(chunk_size=8<<20):
                if not chunk: continue
                output.write(chunk);h.update(chunk);total+=len(chunk)
                if total>=next_report:
                    print(json.dumps({'event':'download','bytes':total,'expected':meta.size}),flush=True)
                    next_report+=128<<20
    if total!=meta.size or h.hexdigest()!=meta.etag:
        raise ValueError(f'checkpoint checksum mismatch after {total} bytes')
    partial.rename(a.output/FILE)
    audit={'status':'complete','repository':REPO,'filename':FILE,'bytes':total,'sha256':h.hexdigest(),
           'revision':meta.commit_hash,'source':'user-authorized Hugging Face signed CDN URL'}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'event':'complete','bytes':total,'output':str(a.output/FILE)}),flush=True)


if __name__=='__main__':
    main()
