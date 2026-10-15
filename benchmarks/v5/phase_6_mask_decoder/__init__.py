"""Phase 6 decoders, organized by research workflow with stable module names.

Public imports and ``python -m`` commands remain
``benchmarks.v5.phase_6_mask_decoder.<module>``. The child directories group
source files; they do not define a second supported import namespace.
"""
from pathlib import Path

_MODULE_GROUPS = {
    'iterative': [
        'episodes', 'iterative_decoder', 'interactive', 'whole_slide_interactive',
        'prepare_spatial_routing', 'train_iterative', 'evaluate_iterative',
    ],
    'boundary': ['boundary_data', 'boundary_refinement', 'probe_boundary_refinement'],
    'dense': [
        'dense_prompt_decoder', 'train_dense_boundary', 'continue_dense_boundary',
        'multiscale_dense_decoder', 'prepare_multiscale_cache', 'train_multiscale_dense',
    ],
    'point': [
        'point_boundary_detail', 'point_objective',
        'train_point_boundary', 'train_point_objective',
    ],
    'diagnostics': [
        'audit_boundary_capacity', 'report_dense_boundary', 'visualize_boundary_refinement',
        'visualize_test_boundary',
    ],
}

_SOURCE_ROOT = Path(__file__).parent
# Python's package search path keeps old imports/CLI and checkpoint class paths
# valid without duplicating implementation files or adding per-file wrappers.
__path__ = [str(_SOURCE_ROOT), *(str(_SOURCE_ROOT / group) for group in _MODULE_GROUPS)]
_MODULE_DIRECTORIES = {name: group for group, names in _MODULE_GROUPS.items() for name in names}


def module_source(name: str) -> Path:
    """Locate a registered module for source hashing after directory migration."""
    return _SOURCE_ROOT / _MODULE_DIRECTORIES[name] / (name + '.py')
