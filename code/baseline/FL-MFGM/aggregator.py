"""FL-MFGM peer-gradient compression with momentum correction."""

import math

import torch

from core_utils.common_utils import clone_state


_LOCAL_PREFIXES = ("user_factors.",)
_PEER_PREFIXES = ("peer_factors.",)


def _has_prefix(key, prefixes):
    return any(key.startswith(prefix) for prefix in prefixes)


def _topk(tensor, ratio):
    """Keep the largest-magnitude ``ratio`` entries of a dense tensor."""
    flat = tensor.flatten()
    count = max(1, min(flat.numel(), math.ceil(flat.numel() * ratio)))
    if count == flat.numel():
        return tensor.clone()
    indices = flat.abs().topk(count, sorted=False).indices
    sparse = torch.zeros_like(flat)
    sparse[indices] = flat[indices]
    return sparse.reshape_as(tensor)


class FLMFGMAggregator:
    """Aggregate compressed peer-factor deltas and retain private user factors."""

    def __init__(self, args):
        self.upload_ratio = float(args.flmfgm_upload_ratio)
        self.momentum_coefficient = float(args.flmfgm_momentum)
        self.momentum = {}
        self.residual = {}
        self.last_upload_bytes = 0
        self.last_upload_nnz = 0

    @staticmethod
    def downlink_bytes(state):
        """Count only the shared peer factors broadcast by the server."""
        return sum(
            value.numel() * value.element_size()
            for key, value in state.items()
            if _has_prefix(key, _PEER_PREFIXES)
        )

    def _compressed_delta(self, client_id, key, state, reference):
        identifier = (int(client_id), key)
        raw = reference[key].float() - state[key].float()
        previous = self.momentum.get(identifier, torch.zeros_like(raw))
        momentum = self.momentum_coefficient * previous + raw
        accumulated = self.residual.get(identifier, torch.zeros_like(raw)) + momentum
        compressed = _topk(accumulated, self.upload_ratio)
        self.momentum[identifier] = momentum.detach()
        self.residual[identifier] = (accumulated - compressed).detach()
        return compressed

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del sample_sizes
        if not states:
            raise ValueError("FL-MFGM requires at least one client state")

        compressed_by_client = []
        self.last_upload_bytes = 0
        self.last_upload_nnz = 0
        for client_id, state in zip(client_ids, states):
            compressed = {}
            for key, value in state.items():
                if _has_prefix(key, _PEER_PREFIXES) and value.is_floating_point():
                    update = self._compressed_delta(
                        client_id, key, state, reference
                    )
                    compressed[key] = update
                    nonzero = int(torch.count_nonzero(update))
                    self.last_upload_nnz += nonzero
                    # Sparse COO estimate: one int64 index and one value.
                    self.last_upload_bytes += nonzero * (8 + value.element_size())
            compressed_by_client.append(compressed)

        global_state = clone_state(reference)
        for key, value in reference.items():
            if _has_prefix(key, _PEER_PREFIXES) and value.is_floating_point():
                mean_delta = torch.stack(
                    [compressed[key].float() for compressed in compressed_by_client]
                ).mean(dim=0)
                global_state[key] = (value.float() - mean_delta).to(value.dtype)

        personalized = {}
        for client_id, state in zip(client_ids, states):
            result = clone_state(global_state)
            for key, value in state.items():
                if _has_prefix(key, _LOCAL_PREFIXES):
                    result[key] = value.clone()
            personalized[client_id] = result
        return global_state, personalized
