"""Verify official medical preprocessing and native point/candidate route."""
import numpy as np
import torch
from torch.nn import functional as F
from benchmarks.v5.fair_comparison.adapted_models import points
from benchmarks.v5.wsi_quality_study.prepare import dump


@torch.no_grad()
def verify_native(model, data, task, output):
    x = data.model_inputs(task, 'xcell', str(next(model.parameters()).device))
    point = x['prompts']
    image = model.rgb(x['support']['image'])[0].byte().permute(1, 2, 0).cpu().numpy()
    expected = model.backend.predict(image, np.asarray(task['positive_xy'], np.float32)*512,
                                    np.asarray(task['negative_xy'], np.float32)*512, 'point', None)
    se = model.encode(x['support'])
    sparse, dense = model.sam.prompt_encoder(points=points(point, size=model.sam.image_encoder.img_size),
                                             boxes=None, masks=None)
    low, quality = model.sam.mask_decoder(image_embeddings=se[0], image_pe=model.sam.prompt_encoder.get_dense_pe(),
                                         sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                         multimask_output=True)
    size = model.sam.image_encoder.img_size
    low = F.interpolate(low, size=(size, size), mode='bilinear', align_corners=False)
    low = F.interpolate(low, size=(512, 512), mode='bilinear', align_corners=False)
    selected = int(quality[0].argmax())
    probability = low[0, selected].sigmoid().cpu().numpy()
    assert selected == expected['candidates']['selected_index']
    # Backend computes sigmoid on CPU with NumPy; compare its rounding explicitly.
    error = float(np.abs(probability-expected['probability']).max())
    assert error <= 2e-6
    assert np.array_equal(probability >= .5, expected['binary_mask'])
    dump(output, dict(status='accepted_official_native_medical_route', selected=selected,
                      binary_exact=True, max_CPU_GPU_sigmoid_difference=error,
                      image_size=size, no_GT_candidate_selection=True))
