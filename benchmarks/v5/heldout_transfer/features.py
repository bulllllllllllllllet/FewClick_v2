"""Predeclared encoder-feature retrieval wrappers; no target/query labels."""
import torch
import torch.nn.functional as F


def sam_features(model,raw):
    rgb=raw['rgb'][0].permute(1,2,0).numpy()
    image=torch.as_tensor(model.transform.apply_image(rgb)).permute(2,0,1)
    return model.sam.image_encoder(model.sam.preprocess(image)[None])


def wsi_features(model,raw):
    features=[]
    for key in ('rgb','rgb_1'):
        image=F.interpolate(raw[key].float(),size=(1024,1024),mode='bilinear',align_corners=False).round().clamp(0,255).to(torch.uint8)
        embedding,_=model.sam.image_encoder(model.sam.preprocess(image[0])[None])
        features.append(embedding.float())
    fine,low=features;h,w=fine.shape[-2:]
    yy,xx=torch.meshgrid((torch.arange(h)+.5)/h,(torch.arange(w)+.5)/w,indexing='ij')
    # Fine [0,1] lies at [0.25,0.75] of its concentric 2x view.
    grid=torch.stack([xx-.5,yy-.5],-1)[None]
    aligned=F.grid_sample(low,grid,align_corners=False)
    return torch.cat([F.normalize(fine,dim=1),F.normalize(aligned,dim=1)],1)
