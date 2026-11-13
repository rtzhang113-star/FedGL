"""Personalized federated aggregation for FNCF."""

import torch

from core_utils.common_utils import clone_state


_LOCAL_PREFIXES = ("user_embedding.", "context_embedding.")


def _is_local(key):
    return any(key.startswith(prefix) for prefix in _LOCAL_PREFIXES)


class FNCFAggregator:
    """FedAvg shared peer/MLP parameters while retaining private user context."""

    def __init__(self, args):
        del args
        self.last_upload_bytes = 0

    @staticmethod
    def downlink_bytes(state):
        """Count only peer/MLP parameters shared with a client."""
        return sum(
            value.numel() * value.element_size()
            for key, value in state.items()
            if not _is_local(key)
        )

    def aggregate(self, states, client_ids, sample_sizes, reference):
        if not states:
            raise ValueError("FNCF requires at least one client state")
        weights = torch.as_tensor(sample_sizes, dtype=torch.float32)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        weights = weights / weights.sum()
        self.last_upload_bytes = sum(
            self.downlink_bytes(state) for state in states
        )

        global_state = clone_state(reference)
        for key, value in reference.items():
            if _is_local(key):
                continue
            if value.is_floating_point():
                global_state[key] = sum(
                    weight * state[key].float()
                    for weight, state in zip(weights, states)
                ).to(value.dtype)
            else:
                global_state[key] = states[int(weights.argmax())][key].clone()

        personalized = {}
        for client_id, state in zip(client_ids, states):
            result = clone_state(global_state)
            for key, value in state.items():
                if _is_local(key):
                    result[key] = value.clone()
            personalized[client_id] = result
        return global_state, personalized
