"""Server-side high-order neighbor selection for HC-FGNN."""

import math

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


def _multi_head_similarity(features, heads):
    """Average scaled dot-product attention over feature subspaces."""
    features = features.detach().float().cpu()
    width = math.ceil(features.shape[1] / heads)
    padded = heads * width - features.shape[1]
    if padded:
        features = F.pad(features, (0, padded))
    features = features.reshape(features.shape[0], heads, width).transpose(0, 1)
    features = F.normalize(features, p=2, dim=2, eps=1e-12)
    scores = torch.matmul(features, features.transpose(1, 2))
    return torch.softmax(scores / math.sqrt(max(width, 1)), dim=2).mean(0)


class HCFGNNAggregator:
    """Select top-k high-order collaborators and distribute neighbor embeddings."""

    def __init__(self, args):
        self.neighbours = int(args.hcfgnn_neighbors)
        self.heads = int(args.hcfgnn_attention_heads)
        self.last_graph = None

    def _graph(self, states):
        client_features = torch.stack(
            [
                state["user_embeds.weight"].detach().float().mean(0).cpu()
                for state in states
            ]
        )
        attention = _multi_head_similarity(client_features, self.heads)
        count = min(max(self.neighbours, 1), len(states))
        values, indices = attention.topk(count, dim=1)
        graph = torch.zeros_like(attention)
        graph.scatter_(1, indices, values)
        graph = graph / graph.sum(1, keepdim=True).clamp_min(1e-12)
        self.last_graph = graph.detach().clone()
        return graph

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del reference
        if not states:
            raise ValueError("HC-FGNN requires at least one client state")
        global_state = _weighted_average(states, sample_sizes)
        graph = self._graph(states)
        user_stack = torch.stack(
            [state["user_embeds.weight"].detach().float() for state in states]
        )
        neighbour_tables = torch.tensordot(graph, user_stack, dims=([1], [0]))

        personalized = {}
        for position, client_id in enumerate(client_ids):
            state = clone_state(global_state)
            # The user's own representation remains local; selected high-order
            # collaborators are delivered through a separate frozen table.
            state["user_embeds.weight"] = states[position][
                "user_embeds.weight"
            ].clone()
            state["neighbour_embeds.weight"] = neighbour_tables[position].to(
                dtype=states[position]["neighbour_embeds.weight"].dtype
            )
            personalized[client_id] = state
        return global_state, personalized
