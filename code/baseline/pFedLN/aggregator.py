"""Layer-wise and neighbor-based personalized aggregation for pFedLN."""

import torch
from torch.nn import functional as F

from core_utils.common_utils import clone_state


def _weighted_average(states, sample_sizes):
    weights = torch.as_tensor(sample_sizes, dtype=torch.float32)
    if float(weights.sum()) <= 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()
    result = {}
    largest = int(weights.argmax())
    for key in states[0]:
        if states[0][key].is_floating_point():
            result[key] = sum(
                weight * state[key] for weight, state in zip(weights, states)
            )
        else:
            result[key] = states[largest][key].clone()
    return result


def _is_personal_layer(key):
    """Map the source implementation's local first/last layers by name."""
    normalized = key.casefold()
    return (
        normalized.startswith("user_embeds.")
        or "head" in normalized
        or "predict" in normalized
    )


def _base_vector(state, max_size):
    keys = [
        key
        for key, value in state.items()
        if value.is_floating_point() and not _is_personal_layer(key)
    ]
    if not keys:
        keys = [key for key, value in state.items() if value.is_floating_point()]
    vector = torch.cat([state[key].detach().float().flatten().cpu() for key in keys])
    if max_size > 0 and vector.numel() > max_size:
        vector = F.adaptive_avg_pool1d(
            vector.reshape(1, 1, -1),
            max_size,
        ).flatten()
    return vector


class PFedLNAggregator:
    """Adapt the supplied pFedLN source to the shared FedGL backbone.

    User embeddings and task heads remain local.  All intervening representation
    layers are aggregated from the top-k most similar participating clients.
    """

    def __init__(self, args):
        self.neighbours = int(args.pfedln_neighbors)
        self.temperature = float(args.pfedln_temperature)
        self.mix = float(args.pfedln_mix)
        self.max_vector_size = int(args.pfedln_vector_size)
        self.last_graph = None

    def _neighbour_graph(self, states):
        vectors = torch.stack(
            [_base_vector(state, self.max_vector_size) for state in states]
        )
        vectors = F.normalize(vectors, p=2, dim=1, eps=1e-12)
        scores = vectors @ vectors.T
        count = min(max(self.neighbours, 1), len(states))
        top_values, top_indices = scores.topk(count, dim=1)
        graph = torch.zeros_like(scores)
        weights = torch.softmax(
            top_values / max(self.temperature, 1e-6),
            dim=1,
        )
        graph.scatter_(1, top_indices, weights)
        self.last_graph = graph.detach().clone()
        return graph

    def _personalize(self, states, graph):
        personalized = [clone_state(state) for state in states]
        for key in states[0]:
            if _is_personal_layer(key):
                continue
            if states[0][key].is_floating_point():
                stacked = torch.stack([state[key].float() for state in states])
                mixed = torch.tensordot(graph, stacked, dims=([1], [0]))
                for client_index in range(len(states)):
                    local = states[client_index][key].float()
                    value = self.mix * mixed[client_index] + (1 - self.mix) * local
                    personalized[client_index][key] = value.to(
                        dtype=states[client_index][key].dtype
                    )
            else:
                for client_index, row in enumerate(graph):
                    source = int(row.argmax())
                    personalized[client_index][key] = states[source][key].clone()
        return personalized

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del reference
        if not states:
            raise ValueError("pFedLN requires at least one client state")
        graph = self._neighbour_graph(states)
        personalized = self._personalize(states, graph)
        global_state = _weighted_average(states, sample_sizes)
        return global_state, {
            client_id: state
            for client_id, state in zip(client_ids, personalized)
        }
