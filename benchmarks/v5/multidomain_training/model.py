"""Differentiable image/cell/support/query bridge with strict retained weights."""
import json
from pathlib import Path

import pyvips
import torch
import torch.nn.functional as F

from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import assignment_geometry,remap_prompt_tokens
from benchmarks.v5.phase_6_mask_decoder.boundary_data import load_baseline
from benchmarks.v5.phase_6_mask_decoder.continue_dense_boundary import DenseContextModel
from benchmarks.v5.phase_6_mask_decoder.evaluate_dense_transfer import DENSE_RUN
from benchmarks.v5.phase_6_mask_decoder.support_query_dense import query_features
from benchmarks.v5.multidomain_data.prepare import sha


def pixel_prompt_tokens(encoded,xy,valid):
    weights=sample_assignment_at_cells(encoded['assignment_low'].float(),xy.float())
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
    return torch.einsum('bkp,bkd->bpd',weights,encoded['tokens'].float())*valid[:,:,None]


class DynamicFewClick(torch.nn.Module):
    def __init__(self,j5,dense,mapper='pixel'):
        super().__init__();self.j5=j5;self.dense=dense;self.mapper=mapper
        if mapper not in ('pixel','centroid'):raise ValueError('unsupported prompt mapping')
        if j5.cross_scale_variant!='fine_only' or j5.parent_context is not None:raise ValueError('requires retained fine_only J5')
        self.joint=False;self.set_stage(False)

    def set_stage(self,joint):
        self.joint=bool(joint);self.j5.requires_grad_(False);self.dense.requires_grad_(True)
        self.j5.cell.requires_grad_(True);self.j5.decoder.requires_grad_(True)
        if joint:
            for module in [self.j5.phase2.embedding,self.j5.phase2.assignment,self.j5.phase2.backbone.layer3,self.j5.phase2.backbone.layer4]:module.requires_grad_(True)

    def train(self,mode=True):
        self.training=mode;self.j5.eval();self.dense.train(mode)
        self.j5.cell.train(mode);self.j5.decoder.train(mode);self.j5.decoder.prompt_model.train(mode)
        # All backbone/embedding/semantic BN running statistics stay frozen.
        self.j5.phase2.eval()
        return self

    def encode_image(self,inputs):
        shallow={}
        hook=self.j5.phase2.backbone.layer1.register_forward_hook(lambda m,i,o:shallow.update(value=o))
        try:features=self.j5.phase2.extract_backbone_features(inputs['image'])
        finally:hook.remove()
        p2=self.j5.phase2.forward_from_features(features,inputs['image'].shape[-2:],return_full_assignment=False,return_tokens=True)
        mass=sample_assignment_at_cells(p2['assignment_low'],inputs['cells'])
        tokens=self.j5.cell(p2['region_tokens'],mass,inputs['cells'],inputs['cell_valid'],inputs['total_cell_count'].float())['fused_tokens']
        xy,area=assignment_geometry(p2['assignment_low'])
        value=shallow['value'];value=value.reshape(value.shape[0],32,8,*value.shape[-2:]).mean(2)
        return dict(assignment_low=p2['assignment_low'],tokens=tokens,xy=xy,area=area,active=area>1e-6,
            deep=p2['embedding'].half(),shallow=value.half(),semantic=F.interpolate(p2['semantic_logits'],size=(64,64),mode='bilinear',align_corners=False).half())

    def encode_support(self,encoded,prompts):
        values={'prompt_size_id':prompts['prompt_size_id']}
        for sign in ['positive','negative']:
            xy,valid=prompts[sign+'_xy'],prompts[sign+'_mask']
            tokens=pixel_prompt_tokens(encoded,xy,valid) if self.mapper=='pixel' else remap_prompt_tokens(encoded['tokens'],encoded['xy'],xy,valid)[0]
            values.update({sign+'_tokens':tokens,sign+'_xy':xy,sign+'_mask':valid})
        return self.j5.decoder.prompt_model.encode_prompt_task(values)

    def forward(self,support,query,prompts,local,shared,with_support=True):
        se=self.encode_image(support);qe=se if shared else self.encode_image(query)
        task=self.encode_support(se,prompts)
        features=query_features(self.j5,qe,task,prompts,query_id='same' if local else 'remote',support_id='same')
        output=self.dense(features)
        source_features=features if shared else query_features(self.j5,se,task,prompts,query_id='same',support_id='same') if with_support else None
        source_output=output if shared else self.dense(source_features) if with_support else None
        return dict(query=output,support=source_output,query_prompt_maps=features['prompt_maps'],task_token=task['task_token'])

    def parameter_groups(self):
        stage=self.joint;self.set_stage(True)
        backbone_ids={id(p) for m in [self.j5.phase2.backbone.layer3,self.j5.phase2.backbone.layer4] for p in m.parameters()}
        head=[p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids]
        backbone=[p for p in self.parameters() if p.requires_grad and id(p) in backbone_ids]
        self.set_stage(stage)
        return [dict(params=head,lr=1e-4,name='heads'),dict(params=backbone,lr=1e-5,name='backbone')]


def load_model(device,mapper='pixel'):
    run=json.loads((DENSE_RUN/'run.json').read_text());checkpoint=Path(run['checkpoint'])
    metadata=json.loads((checkpoint.parent/'run_metadata.json').read_text())
    j5=load_baseline(metadata,checkpoint,device)
    dense=DenseContextModel().to(device)
    payload=torch.load(DENSE_RUN/'dense128_epoch_031.pth',map_location=device,weights_only=False)
    dense.load_state_dict(payload['model'],strict=True)
    model=DynamicFewClick(j5,dense,mapper).to(device).eval()
    dense_path=DENSE_RUN/'dense128_epoch_031.pth'
    return model,dict(j5_checkpoint=str(checkpoint),j5_sha256=sha(checkpoint.read_bytes()),dense_checkpoint=str(dense_path),dense_sha256=sha(dense_path.read_bytes()),mapper=mapper)
