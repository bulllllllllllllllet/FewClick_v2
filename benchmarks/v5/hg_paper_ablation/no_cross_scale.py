"""Remove both region cross-scale message routes, keeping three views and H intact.
Decoder math copied verbatim from architecture_bridge/deterministic_graph.py,
except cross-scale decoder adjacency is omitted. Original source stays frozen.
"""
import torch
import torch.nn.functional as F
from benchmarks.v5.architecture_bridge.deterministic_graph import PaperFewClick,PaperHierarchy,edge_mask
from benchmarks.v4.phase_6_mask_decoder.src.model import knn_adjacency,project_region_probabilities
from benchmarks.v5.phase_6_mask_decoder.dense_prompt_decoder import prompt_maps
class IdentityHierarchy(PaperHierarchy):
    def forward(self,tokens,fm,mc):
        return tokens
class NoCrossScaleCore(PaperFewClick):
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
        if False:  # Only same-scale decoder adjacency remains.
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

