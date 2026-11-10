"""Original objective plus fixed0.5 supervision of the existing region outputs."""
from benchmarks.v5.architecture_bridge.restored_loss import episode_loss as original_loss

AUXILIARY_WEIGHT = 0.5


def episode_loss(output, data, plan, prompts):
    final = original_loss(output, data, plan, prompts)
    region = dict(output, query=output['region_query'], support=output['region_support'])
    # Reuse identical mask/remote-support/click terms. Do not duplicate assignment regularization.
    auxiliary = original_loss(region, data, plan, prompts) - output['assignment_regularization']
    return final + AUXILIARY_WEIGHT * auxiliary
