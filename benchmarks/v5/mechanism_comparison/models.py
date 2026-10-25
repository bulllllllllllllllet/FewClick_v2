"""Fresh, explicitly retrained feature and capacity controls."""
from pathlib import Path

import torch
from torch import nn

from benchmarks.v5.fair_comparison.adapted_models import clean_fewclick
from benchmarks.v5.fair_comparison.audit import digest
from benchmarks.v5.fair_comparison.local_data import fewclick_inputs


VARIANTS={
    'm3_none':dict(multiscale=True,dense=False,cell='none'),
    'm3_morphology':dict(multiscale=True,dense=False,cell='morphology'),
    'm3_xcell':dict(multiscale=True,dense=False,cell='xcell'),
    'm3_shared_encoder':dict(multiscale=True,dense=False,cell='xcell',shared=True),
    'm1_xcell':dict(multiscale=False,dense=False,cell='xcell'),
    'm3_no_hierarchy':dict(multiscale=True,dense=False,cell='xcell',hierarchy=False),
    'm3d_morphology':dict(multiscale=True,dense=True,cell='morphology'),
}


class NoHierarchy(nn.Module):
    def forward(self,tokens,*edges):return tokens


def filter_cells(raw,mode):
    if mode!='morphology':return raw
    result=dict(raw)
    for key,value in raw.items():
        if key=='cells' or key.startswith('cells_'):
            if value.shape[-1]!=74:raise ValueError('morphology control requires audited74-dim inputs')
            result[key]=value.clone();result[key][...,10:]=0
    return result


class MechanismModel(nn.Module):
    cells=True;closed=False
    def __init__(self,variant,device):
        super().__init__();spec=VARIANTS[variant];self.spec=spec;self.contexts=spec['multiscale']
        self.core,self.prior=clean_fewclick(device,'none' if spec['cell']=='none' else 'xcell',spec['multiscale'],spec['dense'])
        if spec.get('shared',False):self.core.encoders=nn.ModuleList([self.core.encoders[0]]*3)
        if not spec.get('hierarchy',True):self.core.hierarchy=NoHierarchy()
        self.prior.update(mechanism_variant=variant,cell_mode=spec['cell'],
            XCell64_used=spec['cell']=='xcell',morphology='same x/y/log-area/unknown identity/count, XCell64 zeroed before fusion' if spec['cell']=='morphology' else None,
            shared_encoder=spec.get('shared',False),hierarchy_enabled=spec.get('hierarchy',True),
            cross_scale_decoder_edges='physical edges retained for all multiscale variants, including no_hierarchy',
            mechanism_sources={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')})
        if spec['cell']=='morphology':self.prior['XCell_overlap']='64-dimensional XCell appearance not used; detection/geometry prior remains'
        if spec['cell']=='xcell':
            self.prior.update(XCell_training_source='internal dataset, not current CRC/TIGER; user confirmed2026-10-25',
                              XCell_overlap='XCell training cohort separately confirmed; upstream CTransPath public image prior recorded independently')
        if not spec['dense']:
            self.prior.update(semantic_channels='unused by region-only decoder',semantic_training='frozen, no semantic supervision')

    def set_stage(self,joint):self.core.set_stage(joint)

    def parameter_groups(self):
        groups=self.core.parameter_groups();seen=set();unique=[]
        for group in groups:
            params=[]
            for parameter in group['params']:
                if id(parameter) not in seen:params.append(parameter);seen.add(id(parameter))
            unique.append(group|dict(params=params))
        return unique

    def forward(self,raw,prompts,**kwargs):
        images=fewclick_inputs(filter_cells(raw,self.spec['cell']))
        output=self.core(images,images,prompts,True,True)
        return output['query']|dict(assignment_regularization=output['assignment_regularization'])


def build(variant,device):
    if variant not in VARIANTS:raise ValueError('unknown mechanism control')
    return MechanismModel(variant,device)
