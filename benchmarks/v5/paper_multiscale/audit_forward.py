"""Real-input checks for context dependence, remote clicks and zero Dense initialization."""
import argparse
import json
from pathlib import Path

import pyvips
import torch

from .data import MultiscaleCorpus,validation_plans
from .model import load_model
from benchmarks.v5.multidomain_data.prepare import dump


def predict(model,inputs):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        return model(**inputs,with_support=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda:0');corpus=MultiscaleCorpus(a.catalog)
    plans=validation_plans(corpus,1);inputs,sup=corpus.materialize(plans[0],device)
    torch.manual_seed(20261025);single,_=load_model(device,False,False)
    first=predict(single,inputs)['query']['pixel_probability']
    altered=dict(inputs);altered['support']=dict(inputs['support']);altered['query']=altered['support']
    for scale in (1,2):altered['support'][f'image_{scale}']=inputs['support'][f'image_{scale}'].flip(-1)
    second=predict(single,altered)['query']['pixel_probability']
    if not torch.equal(first,second):raise ValueError('single-scale unexpectedly depends on context')
    del single;torch.cuda.empty_cache()
    torch.manual_seed(20261025);dense,_=load_model(device,False,True)
    residual=predict(dense,inputs)['query']['pixel_probability']
    error=float((first-residual).abs().max())
    if error>1e-6:raise ValueError('zero Dense initialization changed base prediction')
    del dense;torch.cuda.empty_cache()
    torch.manual_seed(20261025);multi,_=load_model(device,True,False)
    first=predict(multi,inputs)['query']['pixel_probability']
    second=predict(multi,altered)['query']['pixel_probability']
    difference=float((first-second).abs().max())
    if difference<=0:raise ValueError('multiscale does not respond to actual context image changes')
    remote=dict(inputs,local=False,shared=False)
    output=predict(multi,remote)
    if output['query_prompt_maps'].count_nonzero():raise ValueError('source-coordinate prompts leaked into remote query')
    # No supervision object enters model.forward; changing it must leave output exact.
    sup['truth']=~sup['truth'];again=predict(multi,remote)['query']['pixel_probability']
    if not torch.equal(output['query']['pixel_probability'],again):raise ValueError('GT mutation changed forward')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    dump(a.output,dict(passed=True,single_context_invariant=True,multi_context_max_probability_change=difference,
        zero_dense_probability_max_error=error,remote_query_prompt_maps_zero=True,query_GT_independent=True,
        scope='real prepared RGB/cells; untrained common initialization; not performance evaluation'))
    print(a.output.read_text(),flush=True)


if __name__=='__main__':main()
