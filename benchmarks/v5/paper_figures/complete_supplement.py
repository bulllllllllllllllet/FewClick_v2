"""Typeset existing frozen results; no inference or training."""
from pathlib import Path
import json, re, hashlib, statistics

ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'论文/CVPR_HG_mix/CVPR2027_English_1115'
D=json.loads((P/'experimental_table_data.json').read_text())
SOURCE=Path('/nfs-medical3/zyh/fewclick_v2/cvpr_research/WSI_final15_closeout_1113_224245/report.json')
R=json.loads(SOURCE.read_text())

def rows(i):
    return [[c.strip() for c in r.strip('|').split('|')] for r in D[i]['markdown_rows'][2:]]

def table(key,caption,headers,data,wide=False,spec=None):
    env='table*' if wide else 'table'
    spec=spec or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:'+key+'}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}'+spec+'@{}}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(r)+r' \\' for r in data],r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}',''])

methods=['SAM','SAM2','WSI-SAM','SAM-Med2D','SANSA','HG-mix']
ids=[0,1,3,4,5,6,7,8,9,10,11]
names=['Tumor epithelium','Tumor stroma','Necrosis','Normal glands','Normal stroma','Submucosa/serosa','Muscle','Lymphoid aggregates','Mucus','Adipose','Blood']
lookup={(r['method'],r['target'],r['scope']):r for r in R['per_class']}
assert len(lookup)==132
raw={}
for scope,key in [('whole_WSI','wsi_tissue'),('remote_only_WSI','perclass')]:
    output=[]; raw[scope]=[]
    for target,name in zip(ids,names):
        values=[lookup[m,target,scope] for m in methods]
        assert len({v['valid_slides'] for v in values})==1
        scores=[v['Dice']*100 for v in values]; best=max(scores)
        output.append([name,str(values[0]['valid_slides']),*[r'\textbf{'+f'{v:.2f}'+'}' if v==best else f'{v:.2f}' for v in scores]])
        raw[scope].append({'target':target,'tissue':name,'slides':values[0]['valid_slides'],'dice_percent':dict(zip(methods,scores))})
    means=[statistics.mean(lookup[m,t,scope]['Dice'] for t in ids) for m in methods]
    for m,v in zip(methods,means):
        summary=next(r for r in R['summary'] if r['method']==m)
        assert abs(v-summary['whole_Dice' if scope=='whole_WSI' else 'remote_Dice'])<1e-12
    output.append([r'\midrule Macro average','--',*[r'\textbf{'+f'{v*100:.2f}'+'}' if v==max(means) else f'{v*100:.2f}' for v in means]])
    domain='Full-slide' if scope=='whole_WSI' else 'Strict-remote'
    caption=domain+r' Dice (\%) by tissue for all six methods on the same 15 WSIs. Each tissue score averages valid slide--target pairs; $n$ is the number of contributing slides. Macro average weights the 11 tissues equally. Bold marks the highest Dice in each row.'
    if scope!='whole_WSI':caption+=r' Strict remote excludes outputs associated with support-overlapping $4\times$ input footprints.'
    (P/'tables'/f'{key}.tex').write_text(table(key,caption,['Tissue','$n$',*methods[:-1],'FewClick'],output,wide=True))

out={}
data=rows(7)
for r in data:r[0]=r[0].replace('SANSA原生路径','SANSA (native)').replace('SANSA缓存路径','SANSA (cached)')
out['runtime']=table('runtime','Recorded end-to-end WSI workflow times under shared-GPU execution. Model loading and support preparation precede the timed interval. Target counts and concurrent GPU load vary.',['Method/path','Slides','Min./slide'],data)
data=rows(9)
configs=['3 pos., 2 neg.','1 pos., 1 neg.','3 pos., 1 neg.','3 pos., 2 neg.; 2 px','3 pos., 2 neg.; 5 px']
for r,c in zip(data,configs):r[0]=c
out['prompts']=table('prompts',r'Support-click sensitivity on 330 fixed tasks. FC denotes FewClick and SA denotes SANSA. Dice and absent-target FPA are percentages. The final two rows apply radial click perturbations in $512\times512$ support coordinates.', ['Support clicks','FC Dice','SA Dice','FC FPA','SA FPA'],data)
data=rows(10)
for r,c in zip(data,[r'$<1\%$',r'$1$--$5\%$',r'$5$--$20\%$']):r[0]=c
out['size']=table('size',r'Small-target strata in the remote queue. FC: FewClick; SA: SANSA. Dice is class-macro (\%); $\Delta$ is FC minus SA in percentage points.',['Target area','Tasks','FC','SA',r'$\Delta$'],data)
data=rows(11)
for r in data:r[0]={'全体目标存在':'All present','小目标<20%':'Small','大目标≥20%':'Large'}[r[0]]
out['boundary']=table('boundary',r'Coverage and contour metrics (\%) from saved remote predictions. BF1 uses a $2\,\mu$m boundary tolerance. All values are class-macro averages.',['Subset','Method','Prec.','Recall','BF1'],data,spec='llrrr')
data=rows(12)
for r,c in zip(data,['Present Dice','Small Dice','Large Dice','Absent FPA']):r[0]=c
out['bootstrap']=table('bootstrap',r'Paired query-slide bootstrap with fixed model weights. Differences and 95\% intervals are FewClick minus SANSA in percentage points. $B$ is the number of retained replicates.', ['Metric',r'$\Delta$','95\% interval','$B$'],data)
for key,tex in out.items():(P/'tables'/f'{key}.tex').write_text(tex)

f=P/'sec/supplement.tex';s=f.read_text()
for key in ['perclass',*out]:
    s,n=re.subn(r'\\tableplaceholder\{tab:'+key+r'\}[^\n]*',lambda _:r'\input{tables/'+key+'}',s)
    assert n==1 or r'\input{tables/'+key+'}' in s
s=s.replace('This supplement retains implementation details and secondary analyses from the Chinese manuscript. Parameter counts and controlled timing measurements are tabulated below; remaining secondary tables are pending typesetting, with their numerical findings retained in the text.', 'This supplement provides implementation details, complete strict-remote tissue comparisons, timing and memory measurements, support-click sensitivity, size and boundary analyses, and paired query-slide resampling results.')
a=s.index('The following pairs list');b=s.index('\n\n',a)
s=s[:a]+r'Table~\ref{tab:perclass} reports strict-remote Dice for all six methods and all 11 tissues. Full-slide tissue comparisons appear in the main paper. Lymphoid aggregates and blood contribute seven and ten slides; each other tissue contributes 15. Both domains retain the same tissue weighting.'+s[b:]
# Replace numerical prose that duplicates completed tables, while retaining protocol details.
a=s.index('For each method, the five entries');b=s.index('\n\n',a)
s=s[:a]+r'Table~\ref{tab:patchtime} reports all measurements. First-prediction time includes preparation. Cache size counts tensor storage for one support--query feature pair and the first prompt memory. Peak allocated CUDA memory includes model weights and is averaged across tasks.'+s[b:]
a=s.index('With three positive and two negative points');b=s.index('\n\n',a)
s=s[:a]+r'Table~\ref{tab:prompts} compares the five prompt configurations. Three of 990 positive clicks (0.30\%) cross a tissue-label boundary under five-pixel perturbation and are retained according to the fixed rule.'+s[b:]
f.write_text(s)

f=P/'sec/4_experiments.tex';s=f.read_text()
if r'\input{tables/wsi_tissue}' not in s:s=s.replace(r'\input{tables/wsi}',r'\input{tables/wsi}'+'\n'+r'\input{tables/wsi_tissue}')
s=s.replace('per-class results are listed in the supplement.',r'Table~\ref{tab:wsi_tissue} reports full-slide Dice for every tissue and method, with the complete strict-remote breakdown in the supplement.')
s=s.replace('Detailed prompt configurations are reserved in the supplement.','Detailed prompt configurations are tabulated in the supplement.')
f.write_text(s)
(P/'wsi_tissue_data.json').write_text(json.dumps({'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'method_order':methods,'validation':'132/132 unique cells; macro means match report within 1e-12','data':raw},ensure_ascii=False,indent=2)+'\n')
print('Completed six supplementary tables and added six-method full-slide tissue table.')
