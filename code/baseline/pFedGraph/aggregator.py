"""pFedGraph collaboration-graph aggregation adapted from the official code.

The official ICML 2023 implementation builds one collaboration row per client
from the client's data frequency and pairwise model cosine similarity.  The
unconstrained row is projected onto the probability simplex before it is used
to create a personalized aggregate.
"""

import torch
from torch.nn import functional as F

from core_utils.common_utils import clone_state


def _weighted_average(states, sample_sizes):
    weights = torch.as_tensor(sample_sizes, dtype=torch.float32)
    if torch.any(weights < 0):
        raise ValueError("pFedGraph sample sizes must be non-negative")
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


def _project_simplex(vector):
    """Project a finite vector onto {x >= 0, sum(x) = 1}."""
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError("simplex projection expects a non-empty vector")
    if not torch.isfinite(vector).all():
        raise FloatingPointError("pFedGraph graph row contains non-finite values")
    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1
    indices = torch.arange(
        1,
        vector.numel() + 1,
        device=vector.device,
        dtype=vector.dtype,
    )
    support = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(support, as_tuple=False)[-1])
    threshold = cumulative[rho] / indices[rho]
    projected = torch.clamp(vector - threshold, min=0)
    return projected / projected.sum().clamp_min(1e-12)


def _parameter_keys(state, scope):
    floating = [key for key, value in state.items() if value.is_floating_point()]
    if scope == "all":
        return floating
    head = [
        key
        for key in floating
        if "head" in key.casefold() or "predict" in key.casefold()
    ]
    return head or floating


def _flatten_state(state, keys, max_size):
    vector = torch.cat([state[key].detach().float().flatten().cpu() for key in keys])
    if max_size > 0 and vector.numel() > max_size:
        vector = F.adaptive_avg_pool1d(
            vector.reshape(1, 1, -1),
            max_size,
        ).flatten()
    return vector


class PFedGraphAggregator:
    """Official-algorithm-style personalized collaboration graph."""

    def __init__(self, args):
        self.alpha = float(args.pfedgraph_alpha)
        self.scope = args.pfedgraph_similarity_scope
        self.max_vector_size = int(args.pfedgraph_vector_size)
        self.last_graph = None
        self.last_similarity = None

    def _collaboration_graph(self, states, sample_sizes):
        keys = _parameter_keys(states[0], self.scope)
        vectors = torch.stack(
            [
                _flatten_state(state, keys, self.max_vector_size)
                for state in states
            ]
        )
        vectors = F.normalize(vectors, p=2, dim=1, eps=1e-12)
        similarity = (vectors @ vectors.T).clamp(-1, 1)
        frequencies = torch.as_tensor(sample_sizes, dtype=torch.float32)
        if torch.any(frequencies < 0):
            raise ValueError("pFedGraph sample sizes must be non-negative")
        if float(frequencies.sum()) <= 0:
            frequencies = torch.ones_like(frequencies)
        frequencies = frequencies / frequencies.sum()
        graph = torch.stack(
            [
                _project_simplex(
                    frequencies + 0.5 * self.alpha * similarity[row]
                )
                for row in range(len(states))
            ]
        )
        self.last_similarity = similarity.detach().clone()
        self.last_graph = graph.detach().clone()
        return graph

    @staticmethod
    def _personalize(states, graph):
        personalized = [dict() for _ in states]
        for key in states[0]:
            if states[0][key].is_floating_point():
                stacked = torch.stack([state[key].float() for state in states])
                mixed = torch.tensordot(graph, stacked, dims=([1], [0]))
                for client_index in range(len(states)):
                    personalized[client_index][key] = mixed[client_index].to(
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
            raise ValueError("pFedGraph requires at least one client state")
        graph = self._collaboration_graph(states, sample_sizes)
        personalized = self._personalize(states, graph)
        global_state = _weighted_average(states, sample_sizes)
        return global_state, {
            client_id: clone_state(state)
            for client_id, state in zip(client_ids, personalized)
        }
