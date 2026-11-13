"""Compact implementations of the comparison aggregation methods."""

import torch
from torch.nn import functional as F

from core_utils.common_utils import clone_state


PERSONALIZED_METHODS = {
    "local",
    "per",
    "fedper",
    "pfedgraph",
    "pfedln",
    "hcfgnn",
    "flmfgm",
    "fncf",
    "graph",
    "graph_v2",
    "graph_v3",
    "fedgl",
}


def weighted_average(states, sample_sizes):
    weights = torch.tensor(sample_sizes, dtype=torch.float32)
    if float(weights.sum()) == 0:
        weights = torch.ones_like(weights)
    weights = weights / weights.sum()
    result = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            result[key] = sum(
                weight * state[key] for weight, state in zip(weights, states)
            )
        else:
            result[key] = states[0][key].clone()
    return result


def _state_vector(state, reference=None, max_size=2048):
    chunks = []
    for key, value in state.items():
        if value.is_floating_point():
            tensor = value.float()
            if reference is not None:
                tensor = tensor - reference[key].float()
            chunks.append(tensor.flatten())
    vector = torch.cat(chunks)
    if vector.numel() > max_size:
        vector = F.adaptive_avg_pool1d(vector[None, None], max_size).flatten()
    return vector


def cosine_graph(states, reference=None, temperature=0.1):
    features = torch.stack([_state_vector(state, reference) for state in states])
    features = F.normalize(features, dim=1)
    scores = features @ features.T / max(temperature, 1e-6)
    return torch.softmax(scores, dim=1)


def graph_aggregate(states, graph):
    graph = graph.detach().float().cpu()
    result = [dict() for _ in states]
    for key in states[0]:
        if states[0][key].is_floating_point():
            stack = torch.stack([state[key].float() for state in states])
            mixed = torch.tensordot(graph, stack, dims=([1], [0]))
            for index in range(len(states)):
                result[index][key] = mixed[index].to(states[index][key].dtype)
        else:
            for index, state in enumerate(states):
                result[index][key] = state[key].clone()
    return result


class BaselineAggregator:
    def __init__(self, args, prior=None):
        self.args = args
        self.prior = prior

    def _selected_prior(self, client_ids):
        if self.prior is None:
            raise ValueError(f"{self.args.agg} requires a context prior graph")
        ids = torch.tensor(client_ids, device=self.prior.device)
        graph = self.prior.index_select(1, ids).index_select(2, ids).mean(0)
        return graph / graph.sum(1, keepdim=True).clamp_min(1e-12)

    def aggregate(self, states, client_ids, sample_sizes, reference):
        method = self.args.agg
        average = weighted_average(states, sample_sizes)

        if method == "local":
            personalized = [clone_state(state) for state in states]
        elif method == "krum":
            if len(states) <= 2 * self.args.krum_f + 2:
                chosen = average
            else:
                vectors = torch.stack([_state_vector(state) for state in states])
                distances = torch.cdist(vectors, vectors).square()
                neighbours = len(states) - self.args.krum_f - 2
                score = distances.sort(dim=1).values[:, 1 : neighbours + 1].sum(1)
                chosen = states[int(score.argmin())]
            average = clone_state(chosen)
            personalized = [clone_state(average) for _ in states]
        else:
            dynamic = cosine_graph(
                states, reference, self.args.attention_temperature
            )
            if method == "graph":
                graph = self._selected_prior(client_ids).cpu()
            elif method == "graph_v2":
                graph = dynamic
            elif method == "graph_v3":
                prior = self._selected_prior(client_ids).cpu()
                rho = self.args.graph_prior_weight
                graph = (1 - rho) * dynamic + rho * prior
            else:
                raise ValueError(f"unknown aggregation method: {method}")
            graph = graph / graph.sum(1, keepdim=True).clamp_min(1e-12)
            personalized = graph_aggregate(states, graph)

        return average, {
            client_id: state for client_id, state in zip(client_ids, personalized)
        }
