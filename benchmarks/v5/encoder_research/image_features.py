"""Frozen spatial feature contracts; fine512 view only, no target heads."""
import torch
from torch import nn
import torch.nn.functional as F

CHANNELS={'resnet':2048,'phikon':1024,'sam':256}


class FrozenImageEncoder(nn.Module):
    def __init__(self,arm,device='cuda:0'):
        super().__init__();self.arm=arm;self.device_name=device
        if arm=='resnet':
            from torchvision.models.segmentation import deeplabv3_resnet50
            model=deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=True)
            state=torch.load('/nfs-medical3/zyh/v4/phase1/models/deeplabv3_resnet50_coco-cd0a2569.pth',map_location='cpu',weights_only=True)
            self.encoder=model.backbone
            self.encoder.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')},strict=True)
        elif arm=='phikon':
            from transformers import AutoModel,AutoImageProcessor
            path='/nfs-medical3/zyh/fewclick_v2/model_weights/phikon-v2'
            self.encoder=AutoModel.from_pretrained(path,local_files_only=True)
            self.processor=AutoImageProcessor.from_pretrained(path,local_files_only=True)
        elif arm=='sam':
            from segment_anything import sam_model_registry
            model=sam_model_registry['vit_b'](checkpoint='/nfs-medical3/zyh/models/sam/sam_vit_b_01ec64.pth')
            self.encoder=model.image_encoder
        else:raise ValueError(arm)
        self.encoder.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self,rgb):
        assert rgb.ndim==4 and tuple(rgb.shape[1:])==(3,512,512)
        rgb=rgb.to(self.device_name).float()
        if self.arm=='resnet':
            x=rgb/255
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            output=self.encoder(x)['out']
        elif self.arm=='phikon':
            images=[x.permute(1,2,0).byte().cpu().numpy() for x in rgb]
            batch=self.processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values'].to(self.device_name)
            assert tuple(batch.shape[2:])==(512,512)
            output=self.encoder(pixel_values=batch).last_hidden_state[:,1:]
            assert output.shape[1]==32*32
            output=output.transpose(1,2).reshape(len(rgb),1024,32,32)
        else:
            x=F.interpolate(rgb,size=(1024,1024),mode='bilinear',align_corners=False,antialias=True)
            x=(x-x.new_tensor([123.675,116.28,103.53])[None,:,None,None])/x.new_tensor([58.395,57.12,57.375])[None,:,None,None]
            output=self.encoder(x)
        output=F.interpolate(output,size=(32,32),mode='bilinear',align_corners=False)
        assert tuple(output.shape[1:])==(CHANNELS[self.arm],32,32) and torch.isfinite(output).all()
        return output
