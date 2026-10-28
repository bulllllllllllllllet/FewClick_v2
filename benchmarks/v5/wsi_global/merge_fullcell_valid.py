"""Reference accepted immutable real-cell caches in full native grid order."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run(a):
    a.output.mkdir(exist_ok=False, parents=True)
    base, extra = read(a.base/'summary.json'), read(a.extension/'summary.json')
    ba, ea = read(a.base_acceptance/'summary.json'), read(a.extension_acceptance/'summary.json')
    require(ba['status'] == ('accepted_main30_fullcell_cache_gate' if a.fixture else 'accepted_main30_fullcell_cache_qualified')
            and ea['status'] == ('accepted_fullcell_valid_extension_gate' if a.fixture else 'accepted_fullcell_valid_extension_qualified'), 'actual independent real-cell receipts')
    for path, summary, receipt in [(a.base,base,ba),(a.extension,extra,ea)]:
        require(receipt['source_summary_sha256'] == sha(path/'summary.json')
                and summary['cache_manifest_sha256'] == sha(path/'cache_manifest.jsonl'), 'original accepted output bindings')
    require(base['signature']['checkpoint_sha256'] == extra['signature']['checkpoint_sha256']
            and base['signature']['visible_devices'] == extra['signature']['visible_devices'] == '1'
            and base['signature']['weights'] == extra['signature']['weights'], 'same actual model, GPU and detector/feature priors')
    full = a.root/'full_valid_inputs_full_1028_034700'
    require(sha(full/'summary.json') == extra['signature']['input_summary_sha256'], 'same complete native grid')
    alias_accept = a.root/'fullcell_alias_acceptance_1028_233700/summary.json'
    require(read(alias_accept)['status'] == 'accepted_fullcell_same_GPU_full_valid_alias_fixture_only', 'same-GPU original source/query aliases gated on real cells')
    recipe = dict(source_sha256=sha(__file__), input_summary_sha256=sha(full/'summary.json'),
                  alias_fixture_acceptance_sha256=sha(alias_accept), method='fewclick_m3d_xcell')
    if not a.fixture:
        require(a.verified_fixture is not None, 'actual original-order16 index fixture required')
        f = read(a.verified_fixture/'summary.json')
        require(f['status'] == 'passed_fullcell_valid_index_fixture' and f['recipe'] == recipe, 'same merge code/protocol fixture')
    old_entries = list(map(json.loads, (a.base/'cache_manifest.jsonl').open()))
    new_entries = list(map(json.loads, (a.extension/'cache_manifest.jsonl').open()))
    require((len(old_entries),len(new_entries)) == ((8,8) if a.fixture else (20456,7996)), 'complete gate/full input accounting')
    old = {e['identity']['key']:e for e in old_entries}
    fresh = {e['identity']['key']:e for e in new_entries}
    require(len(old)==len(old_entries) and len(fresh)==len(new_entries), 'no duplicate original keys')
    used_old, used_new = set(), set()
    merged = []
    counts = Counter()
    for r in map(json.loads, (full/'inputs.jsonl').open()):
        old_key = r['existing_input_key']
        if old_key in old:
            e = old[old_key]
            used_old.add(old_key)
            origin = 'original_real_cell_cache'
            geometry = e['identity']['geometry']
            require(geometry == dict(native_boxes=r['native_boxes'], native_mpp=r['native_mpp'], source_wsi=r['source_wsi']), 'all original3 real-cell footprints unchanged')
            kind = e['identity']['kind']
            require((kind,r['preparation']) in [('support','support_reuse_candidate'),('query','remote_reuse_candidate')], 'source alias and strict remote role separately preserved')
            counts['support_as_query_alias' if kind=='support' else 'original_remote'] += 1
        elif r['key'] in fresh:
            e = fresh[r['key']]
            used_new.add(r['key'])
            origin = 'new_real_cell_cache'
            require(r['preparation'] == 'new_input' and e['identity']['kind'] == 'full_valid_query'
                and e['identity']['geometry'] == dict(tile=r['tile'], native_boxes=r['native_boxes'], native_mpp=r['native_mpp'], source_wsi=r['source_wsi']), 'truly new physical input with actual cell preparation')
            counts['additional'] += 1
        else:
            require(a.fixture, 'all full28452 fields must have an accepted real-cell cache')
            continue
        d = e['identity']
        require(d['wsi_id'] == r['wsi_id'] and len(d['cell_records']) == len(d['RGB_array_sha256']) == 3, 'all3 actual cell/RGB identities')
        require(Path(e['path']).stat().st_size == e['bytes'] and all(Path(c['path']).is_file() for c in d['cell_records']), 'all immutable accepted artifacts present')
        merged.append(dict(e, full_valid_key=r['key'], original_cache_key=d['key'], cache_origin=origin,
            full_valid_geometry=dict(tile=r['tile'], native_boxes=r['native_boxes'], native_mpp=r['native_mpp'], source_wsi=r['source_wsi'])))
    require(used_old == set(old) and used_new == set(fresh), 'no orphan old/new fields')
    require(len(merged) == len({e['full_valid_key'] for e in merged}) == (16 if a.fixture else 28452), 'complete unique original-order full-grid index')
    require(counts == (dict(original_remote=4,support_as_query_alias=4,additional=8) if a.fixture
                      else dict(original_remote=20208,support_as_query_alias=248,additional=7996)), 'all3 preparation categories retained')
    with (a.output/'cache_manifest.jsonl').open('x') as stream:
        for e in merged:
            stream.write(json.dumps(e)+'\n')
    result = dict(status='passed_fullcell_valid_index_fixture' if a.fixture else 'accepted_fullcell_valid_cache_index_qualified',
        worker_pid=os.getpid(), recipe=recipe, items=len(merged), counts=dict(counts),
        checkpoint_sha256=base['signature']['checkpoint_sha256'], visible_devices='1',
        base=str(a.base), base_summary_sha256=sha(a.base/'summary.json'), base_acceptance=str(a.base_acceptance),
        base_acceptance_sha256=sha(a.base_acceptance/'summary.json'), extension=str(a.extension),
        extension_summary_sha256=sha(a.extension/'summary.json'), extension_acceptance=str(a.extension_acceptance),
        extension_acceptance_sha256=sha(a.extension_acceptance/'summary.json'),
        cache_manifest_sha256=sha(a.output/'cache_manifest.jsonl'),
        scope='Reference index only; original payload/NPZ/encoder identities untouched. All-byte/cell checks inherited from independent base/extension receipts; no fresh all248 support encodes or detector rerun. Full-valid source-inclusive diagnostic separate from strict remote.',
        remaining='Full28452 merged predictions/native output/quality and remote invariance; a16-row fixture does not accept the complete cache.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('root','base','base-acceptance','extension','extension-acceptance','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fixture',action='store_true')
    p.add_argument('--verified-fixture',type=Path)
    run(p.parse_args())
