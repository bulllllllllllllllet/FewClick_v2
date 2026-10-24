"""Independent physical-scale encoders, cell hierarchy, and all-scale graph decoding."""
import copy
import json
from pathlib import Path

import pyvips
import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.v4.phase_3_cell_region.src.model import sample_assignment_at_cells
from benchmarks.v4.phase_6_mask_decoder.src.joint_model import assignment_geometry
from benchmarks.v4.phase_6_mask_decoder.src.model import knn_adjacency, project_region_probabilities
from benchmarks.v4.phase_6_mask_decoder.src.joint_losses import assignment_regularizers
from benchmarks.v5.multidomain_training.model import load_model as load_retained
from benchmarks.v5.multidomain_data.prepare import sha
from benchmarks.v5.phase_6_mask_decoder.dense_prompt_decoder import DensePromptDecoder, prompt_maps


def overlap_edges(child, parent, top_k=4):
    """Detached top-k memberships after exact centered physical resampling."""
    b, k, h, w = child.shape
    yy, xx = torch.meshgrid((torch.arange(h,device=child.device)+.5)/h,
                            (torch.arange(w,device=child.device)+.5)/w,indexing='ij')
    # Parent has twice the field width. Child [0,1] maps into parent [.25,.75].
    grid = torch.stack([xx-.5,yy-.5],dim=-1)[None].expand(b,-1,-1,-1)
    with torch.no_grad():
        sampled = F.grid_sample(parent.detach().float(), grid, align_corners=False)
        overlap = torch.einsum('bkhw,bphw->bkp',child.detach().float(),sampled)
        weights, ids = overlap.topk(min(top_k,parent.shape[1]),dim=-1)
        valid = weights>1e-8
        weights = weights / weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return ids.masked_fill(~valid,-1),weights*valid


def edge_mask(index, n_parent):
    mask = torch.zeros(*index.shape[:2],n_parent,dtype=torch.bool,device=index.device)
    # scatter_add avoids a padded -1 overwriting a real parent-zero edge.
    count = torch.zeros_like(mask,dtype=torch.int32)
    count.scatter_add_(2,index.clamp_min(0),(index>=0).int())
    return count>0


class SparseAttention(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.q=nn.Linear(dim,dim,bias=False)
        self.k=nn.Linear(dim,dim,bias=False); self.v=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,dim)

    def forward(self,query,source,allowed):
        score=torch.einsum('bqd,bkd->bqk',self.q(query),self.k(source))/query.shape[-1]**.5
        valid=allowed.any(-1,keepdim=True)
        score=score.masked_fill(~allowed,torch.finfo(score.dtype).min)
        weights=score.softmax(-1)*allowed.to(score.dtype)
        context=self.out(torch.einsum('bqk,bkd->bqd',weights,self.v(source)))
        return context*valid.to(context.dtype)


class PaperHierarchy(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.up_fm=SparseAttention(dim); self.up_mc=SparseAttention(dim)
        self.down=SparseAttention(dim); self.gate=nn.Parameter(torch.zeros(()))

    def forward(self,tokens,fm,mc):
        fine,middle,coarse=tokens
        m=edge_mask(fm[0],middle.shape[1]); c=edge_mask(mc[0],coarse.shape[1])
        middle=middle+self.up_fm(middle,fine,m.transpose(1,2))
        coarse=coarse+self.up_mc(coarse,middle,c.transpose(1,2))
        ancestor=torch.bmm(m.float(),c.float())>0
        fine=fine+self.gate.tanh()*self.down(fine,torch.cat([middle,coarse],1),torch.cat([m,ancestor],-1))
        return [fine,middle,coarse]


class PaperFewClick(nn.Module):
    def __init__(self,j5,multiscale=True,dense=False):
        super().__init__(); self.multiscale=bool(multiscale); self.use_dense=bool(dense)
        self.encoders=nn.ModuleList([j5.phase2]+[copy.deepcopy(j5.phase2) for _ in range(2 if multiscale else 0)])
        self.cell=j5.cell; self.decoder=j5.decoder
        self.hierarchy=PaperHierarchy(256) if multiscale else None
        self.scale_embedding=nn.Embedding(3,256); nn.init.zeros_(self.scale_embedding.weight)
        # New hierarchy parameter draws must not change the common Dense initialization.
        if dense:
            with torch.random.fork_rng():
                torch.manual_seed(20261025)
                self.dense=DensePromptDecoder(high_resolution=True)
        else:
            self.dense=None
        self.joint=False; self.set_stage(False)

    def set_stage(self,joint):
        self.joint=bool(joint)
        for encoder in self.encoders:
            encoder.requires_grad_(False)
            if joint:
                for module in (encoder.embedding,encoder.assignment,encoder.backbone.layer3,encoder.backbone.layer4):
                    module.requires_grad_(True)
        for module in (self.cell,self.decoder,self.hierarchy,self.scale_embedding,self.dense):
            if module is not None: module.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        for encoder in self.encoders: encoder.eval()  # Freeze all BN statistics.
        # This experiment trains the complete signed set encoder, including attention.
        self.decoder.prompt_model.train(mode)
        return self

    def parameter_groups(self):
        stage=self.joint; self.set_stage(True)
        backbone=[p for e in self.encoders for m in (e.backbone.layer3,e.backbone.layer4) for p in m.parameters()]
        ids={id(p) for p in backbone}
        heads=[p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        self.set_stage(stage)
        return [dict(params=heads,lr=1e-4,name='heads'),dict(params=backbone,lr=1e-5,name='backbone')]

    def encode_image(self,inputs):
        tokens=[]; assignments=[]; geometries=[]; areas=[]; fine_features={}; regularization=[]
        for scale,encoder in enumerate(self.encoders):
            image=inputs['image' if scale==0 else f'image_{scale}']
            cells=inputs['cells' if scale==0 else f'cells_{scale}']
            valid=inputs['cell_valid' if scale==0 else f'cell_valid_{scale}']
            total=inputs['total_cell_count' if scale==0 else f'total_{scale}']
            shallow={}
            hook=encoder.backbone.layer1.register_forward_hook(lambda m,i,o:shallow.update(value=o)) if scale==0 and self.use_dense else None
            try: features=encoder.extract_backbone_features(image)
            finally:
                if hook is not None: hook.remove()
            encoded=encoder.forward_from_features(features,image.shape[-2:],return_full_assignment=False,return_tokens=True)
            assignment=encoded['assignment_low']
            mass=sample_assignment_at_cells(assignment,cells)
            fused=self.cell(encoded['region_tokens'],mass,cells,valid,total.float())['fused_tokens']
            xy,area=assignment_geometry(assignment)
            factor=(1,2,4)[scale]
            # Express every node in the common coarsest field's coordinates.
            geometries.append(.5+(xy-.5)*(factor/4)); areas.append(area*(factor/4)**2)
            tokens.append(fused); assignments.append(assignment)
            regs=assignment_regularizers(assignment)
            regularization.append(.01*regs['assignment_balance']+.001*regs['assignment_entropy']+.01*regs['assignment_compactness'])
            if scale==0:
                fine_features=dict(assignment_low=assignment,local_xy=xy)
                if self.use_dense:
                    v=shallow['value'];v=v.reshape(v.shape[0],32,8,*v.shape[-2:]).mean(2)
                    fine_features.update(deep=encoded['embedding'],shallow=v,
                        semantic=F.interpolate(encoded['semantic_logits'],size=(64,64),mode='bilinear',align_corners=False))
        edges=[]
        if self.multiscale:
            edges=[overlap_edges(assignments[i],assignments[i+1]) for i in (0,1)]
            tokens=self.hierarchy(tokens,*edges)
        return dict(tokens=tokens,xy=geometries,area=areas,edges=edges,
                    regularization=torch.stack(regularization).mean(),**fine_features)

    def support_task(self,encoded,prompts):
        pool=self.decoder.prompt_model; values=dict(prompts); raw=[]; masks=[]
        for sign,identity in [('positive',0),('negative',1)]:
            xy=prompts[sign+'_xy'];valid=prompts[sign+'_mask']
            weights=sample_assignment_at_cells(encoded['assignment_low'].float(),xy.float())
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
            token=torch.einsum('bkp,bkd->bpd',weights,encoded['tokens'][0].float())
            values[sign+'_tokens']=token*valid[:,:,None]
            raw.append(pool.prompt_token_norm(token)+pool.prompt_geometry(xy)+pool.sign_embedding.weight[identity])
            masks.append(valid)
        return pool.encode_prompt_task(values)|dict(raw_prompts=torch.cat(raw,1),raw_valid=torch.cat(masks,1))

    def decode(self,encoded,task,prompts,local):
        pool=self.decoder.prompt_model; sizes=[v.shape[1] for v in encoded['tokens']]
        nodes=torch.cat([v+self.scale_embedding.weight[i] for i,v in enumerate(encoded['tokens'])],1)
        xy=torch.cat(encoded['xy'],1);area=torch.cat(encoded['area'],1);active=area>1e-6
        region=pool.region_norm(nodes)
        context,_=pool.region_prompt_attention(region,task['raw_prompts'],task['raw_prompts'],key_padding_mask=~task['raw_valid'],need_weights=False)
        q=task['task_token']; task_aware=pool.task_aware_norm(region+context+pool.task_projection(q)[:,None])
        sim_pos=F.cosine_similarity(region,task['q_positive'][:,None],dim=-1)
        sim_neg=F.cosine_similarity(region,task['q_negative'][:,None],dim=-1)
        geometry=pool.region_geometry(torch.cat([xy,area[:,:,None]],-1))
        initial=pool.matcher(torch.cat([task_aware,context,geometry,q[:,None].expand(-1,nodes.shape[1],-1),sim_pos[:,:,None],sim_neg[:,:,None]],-1)).squeeze(-1)
        hidden=self.decoder.input_norm(task_aware+self.decoder.logit_embedding(initial[:,:,None])+
            self.decoder.geometry_embedding(torch.cat([xy,area[:,:,None]],-1))+self.decoder.task_projection(q)[:,None])
        adjacency=torch.zeros(nodes.shape[0],nodes.shape[1],nodes.shape[1],device=nodes.device,dtype=torch.bool)
        offset=0
        for size,coords,valid in zip(sizes,encoded['xy'],[v>1e-6 for v in encoded['area']]):
            adjacency[:,offset:offset+size,offset:offset+size]=knn_adjacency(coords,valid,self.decoder.neighbours);offset+=size
        if self.multiscale:
            for child,parent,edge in [(0,1,encoded['edges'][0]),(1,2,encoded['edges'][1])]:
                i=sum(sizes[:child]);j=sum(sizes[:parent]);m=edge_mask(edge[0],sizes[parent])
                adjacency[:,i:i+sizes[child],j:j+sizes[parent]]=m
                adjacency[:,j:j+sizes[parent],i:i+sizes[child]]=m.transpose(1,2)
        for block in self.decoder.blocks: hidden=block(hidden,adjacency)
        # The paper's binary graph-node head; no post-hoc region thresholding.
        logits=self.decoder.delta_head(hidden).squeeze(-1)[:,:sizes[0]]
        assignment=F.interpolate(encoded['assignment_low'],size=(512,512),mode='bilinear',align_corners=False)
        output=project_region_probabilities(assignment,logits)
        maps=prompt_maps(prompts) if local else output['pixel_probability'].new_zeros(nodes.shape[0],2,128,128)
        if self.dense is not None:
            output=self.dense({k:encoded[k] for k in ('deep','shallow','semantic')}|
                              dict(base=output['pixel_probability'],task=q,prompt_maps=maps))
        return output,maps

    def forward(self,support,query,prompts,local,shared,with_support=True):
        se=self.encode_image(support);qe=se if shared else self.encode_image(query)
        task=self.support_task(se,prompts)
        output,maps=self.decode(qe,task,prompts,local)
        source=output if shared else self.decode(se,task,prompts,True)[0] if with_support else None
        regularization=se['regularization'] if shared else .5*(se['regularization']+qe['regularization'])
        return dict(query=output,support=source,query_prompt_maps=maps,task_token=task['task_token'],
                    assignment_regularization=regularization)


def load_model(device,multiscale=True,dense=False):
    retained,weights=load_retained(torch.device('cpu'))
    j5=retained.j5
    # All arms receive identical common initial tensors; Dense starts at zero residual.
    model=PaperFewClick(j5,multiscale,dense).to(device).eval()
    del retained
    return model,dict(j5_checkpoint=weights['j5_checkpoint'],j5_sha256=weights['j5_sha256'],
        multiscale=multiscale,dense=dense,encoders='independent, same pretrained initialization',
        hierarchy='bottom-up attention; gated fine-to-parent-and-ancestor attention',
        decoder='all scales; within-scale KNN and bidirectional parent edges; direct binary head',
        prompt_reader='pixel soft assignment; full signed prompt cross attention',
        dense_initialization='zero residual, no pretrained dense weights',
        historical_initialization_exposure=True,source_sha256=sha(Path(__file__).read_bytes()))
