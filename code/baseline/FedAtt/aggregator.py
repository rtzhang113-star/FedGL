"""Attentive federated aggregation."""

import torch
from torch.nn import functional as F

from core_utils.common_utils import clone_state
from baseline.FedAvg import weighted_average


def _state_vector(state, reference, max_size=2048):
    chunks = [
        (value.float() - reference[key].float()).flatten()
        for key, value in state.items()
        if value.is_floating_point()
    ]
    vector = torch.cat(chunks)
    if vector.numel() > max_size:
        vector = F.adaptive_avg_pool1d(
            vector.reshape(1, 1, -1), max_size
        ).flatten()
    return vector


class FedAttAggregator:
    """Weight client models by their distance to the current server model."""

    def __init__(self, args):
        self.stepsize = float(args.fedatt_stepsize)
        self.last_attention = None

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del sample_sizes
        if not states:
            raise ValueError("FedAtt requires at least one client state")
        vectors = torch.stack(
            [_state_vector(state, reference) for state in states]
        )
        weights = torch.softmax(-vectors.norm(dim=1), dim=0)
        self.last_attention = weights.detach().clone()
        attention = weighted_average(states, weights.tolist())

        global_state = {}
        for key, value in reference.items():
            if value.is_floating_point():
                global_state[key] = (
                    value.float()
                    + self.stepsize * (attention[key].float() - value.float())
                ).to(value.dtype)
            else:
                global_state[key] = attention[key].clone()
        return global_state, {
            client_id: clone_state(global_state) for client_id in client_ids
        }
