"""Physical WSI crops with explicit vendor pitch and fail-fast KFB reads."""
import argparse
import json
import math
from pathlib import Path

import pyvips  # Load before ML libraries in shared experiment processes.
import numpy as np
import pandas as pd
from PIL import Image

from .prepare import digest, write


class StrictSlide:
    def __init__(self, path):
        from vipa_slide_reader import get_slide
        from vipa_slide_reader.kfb.slides import KFBSlide
        self.slide = get_slide(str(path))
        self.kfb = isinstance(self.slide,KFBSlide)
        try:
            self.width,self.height = map(int,self.slide.dimensions)
            props = self.slide.properties
            self.mx = float(props['openslide.mpp-x'])
            if self.kfb:
                # SDK native/src/kfbreader.cpp assigns BOTH keys from the same
                # vendor headerInfo.CapRes. The second key is literally mpp-t.
                self.my = float(props['openslide.mpp-t'])
                if self.mx != self.my:
                    raise ValueError('KFB single CapRes properties disagree')
                self.pitch_source = 'KFB vendor headerInfo.CapRes; SDK mpp-x/mpp-t aliases, isotropic'
            else:
                self.my = float(props['openslide.mpp-y'])
                self.pitch_source = 'OpenSlide mpp-x/mpp-y'
            if min(self.width,self.height)<=0 or not all(math.isfinite(v) and .05<v<5 for v in (self.mx,self.my)):
                raise ValueError('invalid WSI dimensions or physical pitch')
        except Exception:
            self.slide.close()
            raise

    def close(self):
        self.slide.close()

    def read(self, box):
        if len(box)!=4 or any(not isinstance(v,int) for v in box):
            raise ValueError('integer level0 box required')
        x,y,w,h = box
        if min(x,y)<0 or min(w,h)<=0 or x+w>self.width or y+h>self.height:
            raise ValueError('incomplete WSI context; explicit edge protocol required')
        if self.kfb:
            from vipa_slide_reader.kfb.lowlevel import kfbslide_read_roi_region
            # Bypass KFBSlide.read_region's exception-to-white-image behaviour.
            image = kfbslide_read_roi_region(self.slide._osr,0,x,y,w,h)
        else:
            image = self.slide.read_region((x,y),0,(w,h))
        if image.size!=(w,h):
            raise ValueError('reader returned wrong source extent')
        return np.asarray(image.convert('RGB')).copy()

    def fields(self, centre):
        cx,cy = map(float,centre)
        if not np.isfinite([cx,cy]).all():
            raise ValueError('invalid physical centre')
        boxes,images = [],[]
        # Match training's integer fine box, then exact 2x/4x native boxes.
        fw,fh = round(256/self.mx),round(256/self.my)
        for factor in (1,2,4):
            w,h = fw*factor,fh*factor
            box = [round(cx-w/2),round(cy-h/2),w,h]
            native = self.read(box)
            images.append(np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BILINEAR)))
            boxes.append(box)
        return images,dict(native_boxes=boxes,native_mpp=[self.mx,self.my],pitch_source=self.pitch_source,
            field_um=[[w*self.mx,h*self.my] for _,_,w,h in boxes],
            fine_output_mpp=[fw*self.mx/512,fh*self.my/512],padding=False)


def gate(manifest, output):
    output.mkdir(parents=True,exist_ok=False)
    source = pd.read_parquet(manifest).iloc[0].to_dict()
    document = json.loads(Path(source['annotation_json']).read_text())
    polygon = np.asarray([[float(p['x']),float(p['y'])] for p in document[0]['points']])
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
        raise ValueError('invalid representative annotation coordinates')
    centre = (polygon.min(0)+polygon.max(0))/2
    slide = StrictSlide(source['wsi_path'])
    try:
        images, geometry = slide.fields(centre)
        measurements=[]
        for index,image in enumerate(images):
            if image.shape!=(512,512,3) or image.std()<1:
                raise ValueError('representative WSI crop is empty/constant')
            path = output/f'field_{index}.png'
            Image.fromarray(image).save(path)
            if not np.array_equal(np.asarray(Image.open(path)),image):
                raise ValueError('WSI image reload mismatch')
            measurements.append(dict(scale=2**index,image_sha256=digest(path),std=float(image.std()),
                white_fraction=float((image.min(2)>240).mean())))
        rejected=False
        try:
            slide.read([-1,0,128,128])
        except ValueError:
            rejected=True
        if not rejected:
            raise ValueError('invalid context did not fail')
        write(output/'gate.json',dict(status='passed_input_only',manifest_sha256=digest(manifest),
            source_row=0,anonymous_slide_id='tls-'+digest(source['annotation_json'])[:12],
            annotation_sha256=digest(source['annotation_json']),
            dimensions_level0=[slide.width,slide.height],geometry=geometry,images=measurements,
            out_of_bounds_rejected=True,reader_error_policy='propagate; no exception-to-white conversion',
            probe_selection='first manifest row; centre of first annotated polygon bounding box; GT-guided input probe only',
            full_annotation_coverage='not confirmed; no full-slide Dice/FPA',
            complete_grid_upper_bound=math.ceil(slide.width*slide.mx/256)*math.ceil(slide.height*slide.my/256),
            SDK_revision='954175aab19720791db3eb770c1509289e5e4401',
            SDK_native_source_sha256=digest('/home/zhaoyh/vipa-slide-reader/native/src/kfbreader.cpp'),
            source_sha256=digest(__file__),model_run=False,cell_cache_prepared=False))
    finally:
        slide.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    gate(args.manifest,args.output)
