"""Task-conditioned detail refinement at selected uncertain pixels."""
import torch
from torch import nn
import torch.nn.functional as F
from .continue_dense_boundary import DenseContextModel


def sample_features(features, indices, height, width):
    xy=torch.stack(((indices%width+.5)/width,(indices//width+.5)/height),-1)*2-1
    return F.grid_sample(features,xy[:,None].float(),align_corners=False,mode='bilinear')[:,:,0]


def training_indices(probability, truth, valid, strategy, seed, count=4096):
    if strategy not in ('uniform','uncertain'):raise ValueError(strategy)
    device=probability.device;generator=torch.Generator(device=device).manual_seed(seed)
    flat=probability.flatten();target=truth.flatten();mask=valid.flatten()
    positive=torch.where(target&mask)[0];negative=torch.where(~target&mask)[0];all_valid=torch.where(mask)[0]
    if not len(positive) or not len(negative):raise ValueError('binary support required')
    def pick(pool,n):return pool[torch.randint(len(pool),(n,),device=device,generator=generator)]
    anchors=count//8
    balanced=torch.cat((pick(positive,anchors),pick(negative,anchors)))
    remaining=count-len(balanced)
    pool=pick(all_valid,remaining*3)
    if strategy=='uniform':additional=pool[:remaining]
    else:
        uncertain_count=int(remaining*.75)
        chosen=(flat[pool]-.5).abs().topk(uncertain_count,largest=False).indices
        additional=torch.cat((pool[chosen],pick(all_valid,remaining-uncertain_count)))
    return torch.cat((balanced,additional))[None]


class PointBoundaryDetail(nn.Module):
    def __init__(self, coarse_state, inference_points=65536):
        super().__init__()
        self.coarse=DenseContextModel()
        self.coarse.load_state_dict(coarse_state,strict=True)
        self.coarse.requires_grad_(False).eval()
        self.inference_points=inference_points
        self.fine=nn.Sequential(nn.Conv2d(3,24,3,stride=2,padding=1),nn.GroupNorm(6,24),nn.GELU(),
                                nn.Conv2d(24,24,3,padding=1),nn.GELU())
        self.point=nn.Sequential(nn.Conv1d(48+32+24+2+1,64,1),nn.GELU(),nn.Conv1d(64,1,1))
        nn.init.zeros_(self.point[-1].weight);nn.init.zeros_(self.point[-1].bias)
        self.hidden=None
        self.coarse.head.fusion.register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):self.hidden=output

    def train(self,mode=True):
        self.training=mode;self.fine.train(mode);self.point.train(mode);self.coarse.eval()
        return self

    def forward(self,sample,indices=None):
        with torch.no_grad():coarse=self.coarse(sample)['pixel_probability']
        height,width=coarse.shape[-2:]
        if indices is None:
            count=min(self.inference_points,height*width)
            indices=(coarse.flatten(1)-.5).abs().topk(count,largest=False).indices
        fine=self.fine(sample['image'].float())
        features=torch.cat([sample_features(x,indices,height,width) for x in
                            [self.hidden.float(),sample['shallow'].float(),fine,sample['prompt_maps'].float(),coarse[:,None]]],1)
        residual=4*self.point(features)[:,0].tanh()
        coarse_points=coarse.flatten(1).gather(1,indices)
        logits=torch.logit(coarse_points.clamp(1e-6,1-1e-6))+residual
        probability=coarse.clone().flatten(1)
        probability.scatter_(1,indices,logits.sigmoid())
        return {'pixel_probability':probability.reshape_as(coarse),'coarse_probability':coarse,
                'point_logits':logits,'point_probability':logits.sigmoid(),'indices':indices,'residual':residual}
