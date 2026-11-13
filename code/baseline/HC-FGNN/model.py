"""Runnable HC-FGNN-style local QoS predictor.

The paper combines the current user, interacted item, and server-selected
high-order user neighbors through attention, then predicts QoS from the fused
user and item representations.  This implementation extends the scalar paper
prediction to the four normalized FedGL QoS targets.
"""

import math

import torch
from torch import nn

from models.graph_model import _reference_encoder


class HCFGNNModel(nn.Module):
    def __init__(self, args, user_graph, service_graph, num_users, num_services):
        super().__init__()
        self.args = args
        self.num_users = num_users
        self.num_services = num_services
        self.user_graph = user_graph
        self.service_graph = service_graph
        self.use_graph_encoder = bool(args.hcfgnn_use_graph_encoder)
        dimension = args.dimension
        # Client-private graphs include context-only nodes in addition to the
        # QoS matrix rows/columns.  Graph encoding therefore needs embeddings
        # for every graph node; predictions still index only the first
        # num_users/num_services matrix entities.
        user_embedding_count = (
            user_graph.num_nodes() if self.use_graph_encoder else num_users
        )
        service_embedding_count = (
            service_graph.num_nodes() if self.use_graph_encoder else num_services
        )

        self.user_embeds = nn.Embedding(user_embedding_count, dimension)
        self.service_embeds = nn.Embedding(service_embedding_count, dimension)
        self.neighbour_embeds = nn.Embedding(
            user_embedding_count,
            dimension,
            _freeze=True,
        )
        for embedding in (
            self.user_embeds,
            self.service_embeds,
            self.neighbour_embeds,
        ):
            nn.init.xavier_uniform_(embedding.weight)

        # The original adapter was intentionally graph-free.  Keep that mode
        # as the default for reproducibility, while allowing a graph-aware
        # HC-FGNN comparison that uses the same client-private graph inputs as
        # Full-FedGL.  The neighbour table shares the user encoder so that the
        # new branch adds graph context without adding a separate encoder.
        self.user_encoder = None
        self.service_encoder = None
        if self.use_graph_encoder:
            self.user_encoder = _reference_encoder(args, user_graph, num_users)
            self.service_encoder = _reference_encoder(
                args, service_graph, num_services
            )

        self.user_projection = nn.Linear(dimension, dimension, bias=False)
        self.item_projection = nn.Linear(dimension, dimension, bias=False)
        self.neighbour_projection = nn.Linear(dimension, dimension, bias=False)
        self.view_attention = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.LeakyReLU(0.2),
            nn.Linear(dimension, 1, bias=False),
        )
        self.dropout = nn.Dropout(args.dropout)
        interaction_dim = 4 * dimension
        output_dim = 1 if args.task_mode == "single" else 4
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, dimension),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(dimension, output_dim),
        )
        self._cache = None

    def clear_cache(self):
        self._cache = None

    def prepare_eval(self):
        with torch.no_grad():
            self._cache = self.encode()

    def encode(self):
        user = self.user_embeds.weight
        service = self.service_embeds.weight
        neighbour = self.neighbour_embeds.weight
        if self.use_graph_encoder:
            user = self.user_encoder(self.user_graph, user)
            # Neighbour embeddings stay frozen, but their graph-encoded
            # representations still use the locally learned encoder weights.
            neighbour = self.user_encoder(self.user_graph, neighbour)
            service = self.service_encoder(self.service_graph, service)
        return user, service, neighbour

    def forward(self, user_index, service_index, use_cache=False):
        user_nodes, service_nodes, neighbour_nodes = (
            self._cache
            if use_cache and self._cache is not None
            else self.encode()
        )
        user = self.user_projection(user_nodes[user_index])
        item = self.item_projection(service_nodes[service_index])
        neighbour = self.neighbour_projection(neighbour_nodes[user_index])

        views = torch.stack((user, item, neighbour), dim=1)
        anchor = user.unsqueeze(1).expand_as(views)
        attention = torch.softmax(
            self.view_attention(torch.cat((anchor, views), dim=2)).squeeze(2),
            dim=1,
        )
        fused_user = (attention.unsqueeze(2) * views).sum(1)
        fused_user = self.dropout(fused_user)

        interaction = torch.cat(
            (
                fused_user,
                item,
                fused_user * item,
                torch.abs(fused_user - item),
            ),
            dim=1,
        )
        logits = self.prediction_head(interaction)
        # Retain the paper's inner-product signal as a residual prediction.
        dot = (fused_user * item).sum(1, keepdim=True) / math.sqrt(
            fused_user.shape[1]
        )
        output = torch.sigmoid(logits + dot)
        return output.squeeze(1) if self.args.task_mode == "single" else output


def apply_local_dp(parameters, clip, noise_multiplier):
    """Equation (8)-style clipping and adaptive Laplace gradient noise."""
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and parameter.requires_grad
    ]
    if not parameters:
        return
    if clip > 0:
        nn.utils.clip_grad_norm_(parameters, clip)
    if noise_multiplier <= 0:
        return
    with torch.no_grad():
        for parameter in parameters:
            scale = (
                noise_multiplier
                * parameter.grad.detach().abs().mean()
            ).clamp_min(1e-12)
            distribution = torch.distributions.Laplace(
                parameter.grad.new_zeros(()),
                scale,
            )
            parameter.grad.add_(distribution.sample(parameter.grad.shape))
