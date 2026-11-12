"""Opt-in future paired WSI workers; leave active native workers untouched."""
import json
import sys
from pathlib import Path
from benchmarks.v5.whole_slide_once import worker as raster
from benchmarks.v5.encoder_research.continuous_data import sha
from benchmarks.v5.wsi_quality_study.prepare import dump
from .model import CachedSANSA


def main():
    root = Path(sys.argv[sys.argv.index('--root')+1])
    extension = json.loads((root/'cache_extension.json').read_text())
    gate = Path(extension['gate_acceptance'])
    assert sha(gate) == extension['gate_acceptance_sha256']
    assert json.loads(gate.read_text())['status'] == 'accepted_exact_cache_equivalence'
    for path, digest in extension['sources'].items():
        assert sha(path) == digest, path
    original_load, original_infer = raster.load, raster.infer

    def load(method, checkpoint):
        model = original_load(method, checkpoint)
        return CachedSANSA(model).eval() if method == 'sansa' else model

    def infer(model, method, case, supports, out):
        report = original_infer(model, method, case, supports, out)
        if method == 'sansa':
            report.update(encoded_query_reused_across_targets=True, support_memory_reused=True,
                          cache_equivalence_receipt=str(gate),
                          limitations='Exact native/cache equivalence gated; native batch2 duplicated single-frame encoder caches, fixed support memory per target; query memory not accumulated. Streaming only, no persistent whole-slide feature-cache efficiency claim. Historical AAAI test exposure; physical BF1 pending.')
            dump(out/'report.json', report)
            receipt = json.loads((out/'acceptance.json').read_text())
            receipt['report_sha256'] = sha(out/'report.json')
            dump(out/'acceptance.json', receipt)
        return report

    raster.load, raster.infer = load, infer
    raster.main()


if __name__ == '__main__':
    main()
