"""Sample-weighted Federated Averaging."""

import torch

from core_utils.common_utils import clone_state


def weighted_average(states, sample_sizes):
    if not states:
        raise ValueError("FedAvg requires at least one client state")
    weights = torch.as_tensor(sample_sizes, dtype=torch.float32)
    if torch.any(weights < 0):
        raise ValueError("FedAvg sample sizes must be non-negative")
    if float(weights.sum()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()

    result = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            result[key] = sum(
                weight * state[key].float()
                for weight, state in zip(weights, states)
            ).to(value.dtype)
        else:
            result[key] = value.clone()
    return result


class FedAvgAggregator:
    """Aggregate every model parameter according to local sample counts."""

    def __init__(self, args):
        del args

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del reference
        average = weighted_average(states, sample_sizes)
        return average, {
            client_id: clone_state(average) for client_id in client_ids
        }
