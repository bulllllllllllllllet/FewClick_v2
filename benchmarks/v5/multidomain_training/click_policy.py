"""Representative source clicks; whole labelled regions, not random tiny islands."""
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label

from .catalog import interior_labels
from .scoring import BACKGROUND_TARGETS

POLICY='representative_1021_214456'


def _regions(labels, inside, codes, mpp):
    groups=[];height,width=labels.shape
    for code in codes:
        mask=labels==code;components,n=label(mask);areas=np.bincount(components.ravel())
        distance=distance_transform_edt(np.pad(mask,1),sampling=mpp[::-1])[1:-1,1:-1]
        for identity,box in enumerate(find_objects(components),1):
            if box is None:continue
            pixels=np.argwhere((components[box]==identity)&(inside[box]==code))
            if not len(pixels):continue
            pixels+=np.array([box[0].start,box[1].start]);y,x=pixels.T
            clearance=distance[y,x];core=clearance>=max(2.,.5*float(clearance.max()))-1e-6
            y,x=y[core],x[core]
            groups.append(dict(code=int(code),identity=identity,area=int(areas[identity]),
                pixels=np.column_stack((y,x)),clearance=distance[y,x],
                frame=np.minimum.reduce(((x+.5)*mpp[0],(width-x-.5)*mpp[0],(y+.5)*mpp[1],(height-y-.5)*mpp[1]))))
    return groups


def _pick(groups, count, rng, mpp, shape, diverse_labels=False):
    if not groups:raise ValueError('no source interior for requested sign')
    largest=max(g['area'] for g in groups)
    kept=[g for g in groups if g['area']>=.1*largest]
    # If this sign has very little space, explicitly retain other genuine
    # interiors to honour the fixed budget; never invent a pixel or new task.
    limited=sum(len(g['pixels']) for g in kept)<count
    if limited:kept=sorted(groups,key=lambda g:-g['area'])
    if sum(len(g['pixels']) for g in kept)<count:raise ValueError('insufficient representative source interiors')
    pixels=np.concatenate([g['pixels'] for g in kept]);clearance=np.concatenate([g['clearance'] for g in kept])
    frame=np.concatenate([g['frame'] for g in kept]);areas=np.concatenate([np.full(len(g['pixels']),g['area']) for g in kept])
    codes=np.concatenate([np.full(len(g['pixels']),g['code']) for g in kept]);group_ids=np.concatenate([np.full(len(g['pixels']),i) for i,g in enumerate(kept)])
    region_max=np.array([g['clearance'].max() for g in kept]);available=np.ones(len(pixels),bool);chosen=[];used_codes=set()
    spacing=np.full(len(pixels),np.inf)
    for number in range(count):
        eligible=available.copy()
        if number==0:eligible&=areas==largest
        elif diverse_labels and any(c not in used_codes for c in codes[available]):eligible&=~np.isin(codes,list(used_codes))
        # Depth and frame clearance prefer central tissue. Later clicks also
        # cover different places, without blindly favouring tiny components.
        depth=clearance/region_max[group_ids].clip(min=1e-8)
        score=(.25+.75*depth)*np.minimum(frame/8.,1.)*np.sqrt(areas/largest)
        if number:score*=np.minimum(spacing/24.,1.)
        maximum=float(score[eligible].max());candidates=np.flatnonzero(eligible&(score>=.90*maximum-1e-12))
        index=int(rng.choice(candidates));chosen.append(index);available[index]=False;used_codes.add(int(codes[index]))
        delta=(pixels-pixels[index])*np.array(mpp[::-1]);spacing=np.minimum(spacing,np.linalg.norm(delta,axis=1))
    picked=pixels[chosen];height,width=shape
    xy=[[(float(x)+.5)/width,(float(y)+.5)/height] for y,x in picked]
    audit=dict(limited_interior_space=limited,source_components=len(groups),retained_components=len(kept),
        selected_labels=codes[chosen].tolist(),component_area_fraction=(areas[chosen]/largest).tolist(),
        clearance_um=clearance[chosen].tolist(),frame_clearance_um=frame[chosen].tolist(),
        minimum_pair_distance_um=None if count<2 else float(min(np.linalg.norm((picked[i]-picked[j])*np.array(mpp[::-1])) for i in range(count) for j in range(i))))
    return xy,audit


def representative_clicks(labels,target,mpp,positive_count,negative_count,rng,*,dataset):
    """Use only support labels; query image/GT/predictions are not accepted."""
    if dataset not in BACKGROUND_TARGETS or min(positive_count,negative_count)<1:raise ValueError('invalid click contract')
    labels=np.asarray(labels);mpp=np.asarray(mpp,float)
    if labels.ndim!=2 or mpp.shape!=(2,) or not np.isfinite(mpp).all() or (mpp<=0).any():raise ValueError('invalid source geometry')
    inside=interior_labels(labels,mpp)
    positive_groups=_regions(labels,inside,[target],mpp)
    others=[int(c) for c in np.unique(labels) if c not in (target,255)]
    tissue=[c for c in others if c not in BACKGROUND_TARGETS[dataset]]
    negative_groups=_regions(labels,inside,tissue,mpp)
    background_required=sum(len(g['pixels']) for g in negative_groups)<negative_count
    if background_required:negative_groups+=_regions(labels,inside,[c for c in others if c not in tissue],mpp)
    positive,pa=_pick(positive_groups,positive_count,rng,mpp,labels.shape)
    negative,na=_pick(negative_groups,negative_count,rng,mpp,labels.shape,diverse_labels=True)
    return dict(positive_xy=positive,negative_xy=negative),dict(policy=POLICY,positive=pa,negative=na,background_negative_required=background_required)


def repoint(corpus,plan,group=0):
    """New prompts on an immutable task; quality variants share source clicks."""
    source=corpus.index[plan['support_id']];row=corpus.rows[source]
    import json
    rng=np.random.default_rng(corpus.seed+400000000+int(plan['number'])+int(group)*1000000000)
    clicks,audit=representative_clicks(corpus.labels(source),plan['target'],json.loads(row['input_mpp_xy_json']),
        len(plan['positive_xy']),len(plan['negative_xy']),rng,dataset=plan['dataset'])
    return plan|clicks|dict(prompt_policy=POLICY,prompt_group=int(group),click_audit=audit)
