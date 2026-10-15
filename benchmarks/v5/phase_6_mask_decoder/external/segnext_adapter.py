"""Strict pretrained SegNext bridge for the fixed FewClick spatial episodes."""
import copy
from pathlib import Path
import torch
from torch import nn


def rgb_from_normalized(image):
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    return (image*std+mean).clamp(0,1)


def official_points(sample, height=512, width=512):
    """Official DistMaps expects row,col,index; positive and negative equal halves."""
    slots=max(sample['positive_xy'].shape[1],sample['negative_xy'].shape[1],1)
    batch=sample['positive_xy'].shape[0]
    points=sample['positive_xy'].new_full((batch,2*slots,3),-1)
    for group,prefix in enumerate(['positive','negative']):
        xy=sample[prefix+'_xy'];valid=sample[prefix+'_mask'].bool()
        safe=torch.where(valid[...,None],xy,torch.zeros_like(xy))
        x=(safe[...,0].clamp(0,1)*width).floor().clamp(max=width-1)
        y=(safe[...,1].clamp(0,1)*height).floor().clamp(max=height-1)
        index=torch.arange(xy.shape[1],device=xy.device,dtype=xy.dtype)[None].expand(batch,-1)
        packed=torch.stack((y,x,index+group*slots),-1)
        points[:,group*slots:group*slots+xy.shape[1]]=torch.where(valid[...,None],packed,torch.full_like(packed,-1))
    return points


class SegNextAdapter(nn.Module):
    def __init__(self, config, state):
        super().__init__()
        # mmcv-lite is supplied by an explicit isolated PYTHONPATH, not installed
        # into the shared aligner environment; import only when requested.
        from .segnext_vendor.model import PlainVitModel
        if config['class']!='isegm.model.is_plainvit_model.PlainVitModel':
            raise ValueError('unexpected upstream model class')
        if any(v['type']!='builtin' for v in config['params'].values()):
            raise ValueError('non-builtin constructor config')
        self.config=copy.deepcopy(config)
        kwargs={key:value['value'] for key,value in config['params'].items()}
        if kwargs['cpu_dist_maps'] or kwargs['fusion_params']['type']!='self_attention':
            raise ValueError('unsupported pretrained configuration')
        self.model=PlainVitModel(**kwargs)
        self.model.load_state_dict(state,strict=True)
        self.model.backbone.eval().requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path):
        payload=torch.load(Path(path),map_location='cpu',weights_only=False)
        return cls(payload['config'],payload['state_dict'])

    def train(self, mode=True):
        super().train(mode)
        self.model.backbone.eval()
        return self

    def encode(self, sample):
        with torch.no_grad():
            return self.model.get_image_feats(rgb_from_normalized(sample['image'].float()),keep_shape=True)

    def forward(self, sample):
        height,width=sample['base'].shape[-2:]
        self.model.orig_size=(height,width)
        image_features=sample['segnext_features'].float()
        points=official_points(sample,height,width)
        prompts={'points':points,'prev_mask':image_features.new_zeros(image_features.shape[0],1,height,width)}
        prompt_features=self.model.get_prompt_feats((height,width),prompts)
        logits=self.model((height,width),image_features,prompt_features)['instances'][:,0]
        return {'pixel_logits':logits,'pixel_probability':logits.sigmoid()}
