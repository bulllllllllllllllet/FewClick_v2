"""Typeset audited manuscript tables and integrate existing empirical figures (CPU only)."""
from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / '论文/CVPR_HG_mix/CVPR2027_English_1115'
DATA = json.loads((P / 'experimental_table_data.json').read_text())
OUT = P / 'tables'
OUT.mkdir(exist_ok=True)

def rows(index):
    return [[x.strip().replace('FewClick（HG-mix）', 'FewClick') for x in s.strip('|').split('|')]
            for s in DATA[index]['markdown_rows'][2:]]

def table(label, caption, headers, values, spec=None, best=None):
    spec = spec or 'l' + 'r' * (len(headers) - 1)
    body = []
    for row in values:
        cells = list(row)
        if best:
            for col, direction in best.items():
                optimum = direction(float(r[col]) for r in values)
                if float(row[col]) == optimum:
                    cells[col] = r'\textbf{' + row[col] + '}'
        body.append(' & '.join(cells) + r' \\')
    return '\n'.join([
        r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
        r'\label{' + label + '}', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}' + spec + r'@{}}', r'\toprule',
        ' & '.join(headers) + r' \\', r'\midrule', *body,
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])

outputs = {}
outputs['remote'] = table('tab:remote',
    r'Same-slide remote segmentation on 11,000 tasks, with three positive and two negative support clicks and no query clicks. All metrics are percentages; bold marks the best value.',
    ['Method', r'Dice $\uparrow$', r'Large $\uparrow$', r'FPA $\downarrow$'],
    rows(2), best={1:max, 2:max, 3:min})
outputs['wsi'] = table('tab:wsi',
    r'Whole-slide class-macro Dice (\%) on 15 slides and 152 tissue tasks. Each slide--target pair uses one support patch with three positive and two negative clicks. Strict remote excludes support-overlapping input footprints.',
    ['Method', r'Full slide $\uparrow$', r'Strict remote $\uparrow$'], rows(4), best={1:max,2:max})

# The source factorial table reports present/large/FPA, not small-target Dice.
factorial = []
for name, row in zip(['Whole', 'Tiled', 'Fused'], rows(5)):
    for flag, value in zip(['No', 'Yes'], row[1:]):
        factorial.append([name, flag, *[x.strip() for x in value.split('/')]])
ab = table('tab:ablation',
    r'Matched structural ablations: (a) fine representation and dense transfer; (b) components of the fused model. P/L/S denote present-, large-, and small-target Dice; all values are percentages. Higher Dice and lower FPA are better.',
    ['Fine view', 'Transfer', 'P', 'L', 'FPA'], factorial, spec='llrrr')
ab = ab.replace(r'\begin{tabular}', r'\textbf{(a) Fine representation $\times$ dense transfer}\\[2pt]'+'\n'+r'\begin{tabular}',1)
variants = [['Full FewClick',*rows(6)[0][1:]], ['No dense transfer',*rows(6)[1][1:]], ['No region messages',*rows(6)[2][1:]]]
second = '\n'.join([r'\par\vspace{4pt}',r'\textbf{(b) Fused-model components}\\[2pt]',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Variant & P & L & S & FPA \\',r'\midrule',*[' & '.join(r)+r' \\' for r in variants],r'\bottomrule',r'\end{tabular}'])
outputs['ablation'] = ab.replace(r'\end{table}',second+'\n'+r'\end{table}')
outputs['efficiency'] = table('tab:efficiency',
    r'Mean latency and cache size over 100 tasks on one RTX 5880 Ada GPU. First prediction includes reading and encoding; changed-prompt prediction reuses image features. Full preparation and memory measurements are given in the supplement.',
    ['Method', r'First (ms)', r'Changed (ms)', 'Cache (MiB)'],
    [[r[0],r[2],r[3],r[4]] for r in rows(8)])
outputs['protocol'] = table('tab:protocol',
    r'Parameter inventory for evaluated models. Frozen and trainable parameters are included in the total; shared tensors are counted once. Initialization, input views, and adaptation interfaces are specified in the accompanying text.',
    ['Method', 'Total', 'Trainable'], rows(1))
# Full timing table accompanies the compact main-paper table.
full = table('tab:patchtime',
    r'Full timing and memory measurements, averaged over the same 100 tasks. Preparation is included in first-prediction latency. Cache and peak allocated CUDA memory are in MiB; peak memory includes weights.',
    ['Method','Prep. (ms)','First (ms)','Changed (ms)','Cache (MiB)','Peak (MiB)'], rows(8))
outputs['patchtime'] = full.replace(r'\begin{table}',r'\begin{table*}').replace(r'\end{table}',r'\end{table*}')
for name, tex in outputs.items():
    (OUT / (name+'.tex')).write_text(tex)
for filename, names in [('4_experiments.tex',['remote','wsi','ablation','efficiency']),('supplement.tex',['protocol','patchtime'])]:
    path = P/'sec'/filename
    content = path.read_text()
    for name in names:
        content, n = re.subn(r'\\tableplaceholder\{tab:'+name+r'\}[^\n]*',lambda _:r'\input{tables/'+name+'}',content)
        if n != 1 and r'\input{tables/'+name+'}' not in content:
            raise ValueError(name)
    path.write_text(content)

src = ROOT/'论文/CVPR_HG_mix/figures/existing_results_1114_135506'
for name in ['figure2.pdf','figure3.pdf','case_selection.json','wsi_selection.json']:
    shutil.copy2(src/name,P/'figures'/name)
path=P/'sec/4_experiments.tex';text=path.read_text()
text=text.replace(r'\centering\reserved{1.05in}{Whole-slide qualitative comparison reserved}',r'\centering\includegraphics[width=\textwidth]{figures/figure2.pdf}')
text=text.replace(r'\centering\reserved{1.10in}{Remote segmentation and failure-case panels reserved}',r'\centering\includegraphics[width=\textwidth]{figures/figure3.pdf}')
text=text.replace('The reserved panel will show the first slide and first tissue target in the fixed formal manifest (tumor epithelium), the support clicks, valid and strict-remote scoring domains, and six-method predictions.', 'The first slide and first tissue target in the fixed formal manifest (tumor epithelium) are shown with support clicks, scoring domains, and six-method predictions. Green plus and cyan cross markers denote positive and negative clicks; red overlays denote target tissue. Gray and blue together mark the valid full-slide domain, and blue marks the strict-remote subset. Panel scores are full-slide Dice for this slide--target pair.')
text=text.replace('Each reserved row contains','Each row contains').replace('Metrics use the original $512\\times512$ predictions.', 'Green plus and cyan cross markers denote positive and negative support clicks; red overlays denote target tissue. Panel scores are task-level Dice for present targets and FPA for the absent target. Metrics use the original $512\\times512$ predictions.')
text=text.replace('Figure~\\ref{fig:wsi} reserves the corresponding qualitative layout; per-class results are reserved in the supplement.', 'Figure~\\ref{fig:wsi} shows the corresponding qualitative comparison; per-class results are listed in the supplement.')
text=text.replace('Figure~\\ref{fig:failures} reserves four retrospectively selected descriptive cases.', 'Figure~\\ref{fig:failures} shows four retrospectively selected descriptive cases.')
text=text.replace('The supplementary material reserves a full protocol and parameter table.', 'The supplementary material details the comparison protocol and parameter counts.')
path.write_text(text)
path=P/'sec/supplement.tex';text=path.read_text().replace('The reserved main-paper WSI figure','The main-paper WSI figure').replace('The planned panels show','The panels show').replace('The four reserved rows','The four qualitative rows').replace('the reserved display overlays','the display overlays')
text=text.replace('Experimental tables are reserved for subsequent typesetting; the associated numerical findings are retained in the text.', 'Parameter counts and controlled timing measurements are tabulated below; remaining secondary tables are pending typesetting, with their numerical findings retained in the text.')
path.write_text(text)
print('Completed four main tables, two empirical figures, and two supplementary tables.')
