"""User-defined tissue-target task scope; background pixels remain valid negatives."""

POLICY_VERSION='tissue_targets_1021_184142'
BACKGROUND_TARGETS={'crc':(2,),'ignite':(1,),'puma':(6,),'ocelot':(),'tiger':()}


def split_target_tasks(rows, dataset=None):
    """Filter episodes by their positive target, never pixels or target presence."""
    primary=[];excluded=[]
    for row in rows:
        name=row.get('dataset',dataset)
        if name not in BACKGROUND_TARGETS:raise ValueError(f'unknown dataset for scoring: {name}')
        target=row['target'] if 'target' in row else row['target_class']
        (excluded if int(target) in BACKGROUND_TARGETS[name] else primary).append(row)
    return primary,excluded


def scoring_policy():
    return dict(version=POLICY_VERSION,primary_scope='tissue-positive-target episodes',
        excluded_positive_targets={k:list(v) for k,v in BACKGROUND_TARGETS.items() if v},
        background_pixels_are_valid_negatives=True,absent_tissue_queries_are_scored=True,
        historical_all_target_scores_are_archived=True,
        authorization='User2026-10-21: background as a positive target is outside the algorithm task')
