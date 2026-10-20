"""Explicit physical geometry and conservative pixel-center polygon labels."""
import hashlib
import math

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

IGNORE = 255


def stable_bucket(key, seed=20261020):
    return int(hashlib.sha256(f'{seed}|{key}'.encode()).hexdigest()[:16], 16) % 100


def spatial_split(slide, box, block=8192, margin=64):
    """Released TIGER 0.5MPP coordinates; no patient grouping."""
    x, y, w, h = box
    bx, by = math.floor((x+w/2)/block), math.floor((y+h/2)/block)
    if x < bx*block+margin or y < by*block+margin or x+w > (bx+1)*block-margin or y+h > (by+1)*block-margin:
        return None
    group = f'tiger|{slide}|{bx}|{by}'
    value = stable_bucket(group)
    return {'split': 'train' if value < 70 else 'val' if value < 85 else 'test', 'spatial_group_id': group}


def ring_mask(ring, shape):
    """Even-odd fill at pixel centers, half-open edge rule, no new dependency."""
    points = np.asarray(ring, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError('invalid GeoJSON ring')
    points = points[:, :2]
    if not np.array_equal(points[0], points[-1]):
        raise ValueError('GeoJSON ring must be closed')
    h, w = shape
    result = np.zeros(shape, dtype=bool)
    a, b = points[:-1], points[1:]
    for y in range(max(0, math.ceil(points[:,1].min()-.5)), min(h, math.ceil(points[:,1].max()-.5))):
        yy = y+.5
        active = ((a[:,1] <= yy) & (yy < b[:,1])) | ((b[:,1] <= yy) & (yy < a[:,1]))
        aa, bb = a[active], b[active]
        xx = np.sort(aa[:,0] + (yy-aa[:,1])*(bb[:,0]-aa[:,0])/(bb[:,1]-aa[:,1]))
        if len(xx) % 2:
            raise ValueError('odd polygon scan-line crossings')
        for x0, x1 in xx.reshape(-1, 2):
            left, right = max(0, math.ceil(x0-.5)), min(w, math.ceil(x1-.5))
            if right > left:
                result[y, left:right] = True
    return result


def polygon_mask(coordinates, shape):
    if not coordinates:
        raise ValueError('polygon without exterior')
    result = ring_mask(coordinates[0], shape)
    for hole in coordinates[1:]:
        result &= ~ring_mask(hole, shape)
    return result


def rasterize_tissue(document, shape, names):
    """Background has low priority; contradictory specific labels are ignore."""
    result = np.full(shape, IGNORE, np.uint8)
    conflicts = np.zeros(shape, bool)
    background = np.zeros(shape, bool)
    for feature in document['features']:
        name = feature['properties']['classification']['name']
        if name not in names:
            raise ValueError(f'unknown PUMA label: {name}')
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            polygons = [geometry['coordinates']]
        elif geometry['type'] == 'MultiPolygon':
            polygons = geometry['coordinates']
        else:
            raise ValueError(f'unsupported geometry: {geometry["type"]}')
        mask = np.zeros(shape, bool)
        for polygon in polygons:
            mask |= polygon_mask(polygon, shape)
        label = names[name]
        if name == 'tissue_white_background':
            background |= mask
            continue
        conflicts |= mask & (result != IGNORE) & (result != label)
        result[mask & (result == IGNORE)] = label
    result[background & (result == IGNORE)] = names['tissue_white_background']
    result[conflicts] = IGNORE
    return result, {'conflict_pixels': int(conflicts.sum()), 'valid_fraction': float((result != IGNORE).mean())}


def standardize(rgb, labels, box, mpp, size=512):
    """Image and mask cover identical extent, even when their pixel grids differ."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or labels.ndim != 2:
        raise ValueError('expected RGB and categorical GT')
    x, y, w, h = map(int, box)
    if min(w, h, *mpp) <= 0:
        raise ValueError('non-positive extent/MPP')
    # Padding must be white before anti-aliasing as well as after resampling;
    # PIL.crop outside the image would otherwise introduce a dark border inside
    # the valid tissue edge, even if invalid output pixels are later whitened.
    region = Image.new('RGB',(w,h),(255,255,255))
    left,top,right,bottom=max(0,x),max(0,y),min(rgb.shape[1],x+w),min(rgb.shape[0],y+h)
    if right>left and bottom>top:
        region.paste(Image.fromarray(rgb).crop((left,top,right,bottom)),(left-x,top-y))
    image = np.asarray(region.resize((size,size), Image.Resampling.LANCZOS)).copy()
    yy = y+(np.arange(size)+.5)*h/size
    xx = x+(np.arange(size)+.5)*w/size
    in_bounds = (yy[:,None] >= 0) & (yy[:,None] < rgb.shape[0]) & (xx[None,:] >= 0) & (xx[None,:] < rgb.shape[1])
    gy, gx = np.meshgrid(yy*labels.shape[0]/rgb.shape[0]-.5, xx*labels.shape[1]/rgb.shape[1]-.5, indexing='ij')
    gt = map_coordinates(labels, [gy,gx], order=0, mode='nearest', prefilter=False).astype(np.uint8)
    gt[~in_bounds] = IGNORE
    image[~in_bounds] = 255
    metadata = {'input_mpp_xy': [w*mpp[0]/size, h*mpp[1]/size],
                'input_to_image_affine': [w/size,0,x,0,h/size,y],
                'source_image_size': [rgb.shape[1],rgb.shape[0]],
                'geometric_valid_fraction': float(in_bounds.mean())}
    return image, gt, metadata


def geometric_mask(metadata, size=512):
    a=metadata['input_to_image_affine'];w,h=metadata['source_image_size']
    x=(np.arange(size)+.5)*a[0]+a[2];y=(np.arange(size)+.5)*a[4]+a[5]
    return (x[None,:]>=0)&(x[None,:]<w)&(y[:,None]>=0)&(y[:,None]<h)


def quality_version(rgb, input_mpp, available_mpp, quality, metadata=None):
    if quality == 'original':
        return rgb.copy(), {'intermediate_size': list(rgb.shape[:2][::-1]), 'effective_quality_mpp_xy': list(available_mpp)}
    requested = float(quality)
    intermediate = tuple(max(1, min(rgb.shape[1-i], round(rgb.shape[1-i]*input_mpp[i]/max(requested, available_mpp[i])))) for i in range(2))
    image = Image.fromarray(rgb).resize(intermediate, Image.Resampling.LANCZOS).resize((rgb.shape[1],rgb.shape[0]), Image.Resampling.BILINEAR)
    result=np.asarray(image).copy()
    if metadata is not None:
        result[~geometric_mask(metadata,rgb.shape[0])]=255
    return result, {'intermediate_size': list(intermediate),
        'effective_quality_mpp_xy': [rgb.shape[1-i]*input_mpp[i]/intermediate[i] for i in range(2)]}
