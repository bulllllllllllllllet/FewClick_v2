import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import requests

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);r=p.parse_args().root
 vendor=r/'SANSA';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()
 deps=r/'deps';subprocess.run([sys.executable,'-m','pip','install','--target',str(deps),'hydra-core==1.3.2','py3_wget','iopath'],check=True)
 sources={'sam2_hiera_large.pt':'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt','sansa_universal.pth':'https://github.com/ClaudiaCuttano/SANSA/releases/download/v1.0.0/sansa_universal.pth'};weights={};(vendor/'pretrain').mkdir(exist_ok=True)
 for name,url in sources.items():
  path=vendor/'pretrain'/name;assert not path.exists()
  with requests.get(url,stream=True,timeout=(30,120)) as response:
   response.raise_for_status();size=0;last=time.monotonic()
   with path.with_suffix(path.suffix+'.partial').open('xb') as f:
    for chunk in response.iter_content(1024*1024):
     f.write(chunk);size+=len(chunk)
     if time.monotonic()-last>10:print(json.dumps(dict(file=name,bytes=size)),flush=True);last=time.monotonic()
   if response.headers.get('Content-Length'):assert size==int(response.headers['Content-Length'])
  path.with_suffix(path.suffix+'.partial').rename(path);weights[name]=dict(path=str(path),url=url,bytes=size,sha256=digest(path));print(json.dumps(weights[name]),flush=True)
 (r/'weights.json').write_text(json.dumps(dict(source_commit=commit,weights=weights),indent=2)+'\n')
if __name__=='__main__':main()
