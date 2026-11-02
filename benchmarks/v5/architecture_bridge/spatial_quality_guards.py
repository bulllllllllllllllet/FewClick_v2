"""P3 preregistered safety checks, including remote strata and domain FPA."""
from .compare_vocabulary import summarize_arm
from .compare_restored import BOUNDS
from .remote_safety import summarize_remote
from benchmarks.v5.encoder_research.compare_continuous_pilots import resource_guards


def compare_rows(reference,candidate):
    assert len(reference)==len(candidate)==3711
    for a,b in zip(reference,candidate):
        assert a['scope']==b['scope'] and a['budget']==b['budget'] and a['plan']==b['plan']
        for key in ['positive','valid','true_boundary_pixels']:assert a['metrics'][key]==b['metrics'][key]
    _,_,ref=summarize_arm(reference);_,_,cand=summarize_arm(candidate)
    base=resource_guards(ref,cand,BOUNDS)
    r=summarize_remote(reference);c=summarize_remote(candidate);extra={};changes={}
    for name in ['small','fat']:
        x,y=r[name],c[name];assert x['n']==y['n']
        valid=x['mean'] is not None and y['mean'] is not None
        extra['remote_'+name]=bool(valid and y['mean']>=x['mean']-.02)
        changes['remote_'+name]=y['mean']-x['mean'] if valid else None
    for d in ['crc','tiger']:
        x,y=r['per_domain'][d],c['per_domain'][d];assert x['n']==y['n']
        valid=x['mean'] is not None and y['mean'] is not None
        extra[d+'_absent_FPA']=bool(valid and y['mean']<=x['mean']+.01)
        changes[d+'_absent_FPA']=y['mean']-x['mean'] if valid else None
    return dict(original_six=base,remote_safety_tests=extra,remote_changes=changes,
                pass_all=base['pass_all'] and all(extra.values()),reference_remote=r,candidate_remote=c,
                limitation='point guards only; formal run bindings, uncertainty and seed replication required separately')
