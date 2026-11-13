"""Compact dual-branch HAN/GAT predictor used by every federated client."""

import math

import dgl
import dgl.nn as dglnn
from dgl.nn.functional import edge_softmax
import torch
from torch import nn
from torch.nn import functional as F

from graph_construction.graph_builder import projected_entity_graph


class GATEncoder(nn.Module):
    def __init__(self, dimension, heads, dropout):
        super().__init__()
        head_dim = math.ceil(dimension / heads)
        self.gat = dglnn.GATConv(
            dimension,
            head_dim,
            heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            allow_zero_in_degree=True,
        )
        output_dim = head_dim * heads
        self.project = (
            nn.Identity() if output_dim == dimension else nn.Linear(output_dim, dimension)
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, graph, features):
        encoded = self.gat(graph, features).flatten(1)
        return self.norm(F.elu(self.project(encoded)))


class HANEncoder(nn.Module):
    """Two semantic GAT views followed by semantic-level attention."""

    def __init__(self, graph, num_entities, dimension, heads, dropout):
        super().__init__()
        self.graphs = (graph, projected_entity_graph(graph, num_entities))
        self.views = nn.ModuleList(
            [GATEncoder(dimension, heads, dropout) for _ in self.graphs]
        )
        self.semantic = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, 1, bias=False),
        )

    def forward(self, _, features):
        views = [encoder(graph, features) for encoder, graph in zip(self.views, self.graphs)]
        scores = torch.stack([self.semantic(view).mean(0) for view in views])
        weights = torch.softmax(scores, dim=0)
        return sum(weight * view for weight, view in zip(weights, views))


class BipartiteInteractionAttention(nn.Module):
    """Propagate over the client's private user-service interactions."""

    def __init__(self, in_dim, out_dim, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.user_transform = nn.Linear(in_dim, out_dim, bias=False)
        self.service_transform = nn.Linear(in_dim, out_dim, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(2 * out_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_transform.weight)
        nn.init.xavier_uniform_(self.service_transform.weight)
        nn.init.xavier_uniform_(self.attention_vector)

    def _edge_attention(self, edges):
        pair_features = torch.cat((edges.src["h"], edges.dst["h"]), dim=1)
        logits = torch.matmul(pair_features, self.attention_vector).squeeze(-1)
        return {"attention": self.leaky_relu(logits)}

    def forward(self, graph, user_features, service_features):
        expected_nodes = user_features.size(0) + service_features.size(0)
        if graph.num_nodes() != expected_nodes:
            raise ValueError(
                "Interaction graph node count must equal users plus services"
            )
        transformed = torch.cat(
            (
                self.user_transform(user_features),
                self.service_transform(service_features),
            ),
            dim=0,
        )
        with graph.local_scope():
            graph.ndata["h"] = transformed
            graph.apply_edges(self._edge_attention)
            weights = edge_softmax(
                graph, graph.edata["attention"], norm_by="dst"
            )
            graph.edata["attention"] = self.dropout(weights)
            graph.update_all(
                dgl.function.u_mul_e("h", "attention", "m"),
                dgl.function.sum("m", "h_agg"),
            )
            return graph.ndata["h_agg"]


class SemanticFusion(nn.Module):
    """Fuse context and interaction views with semantic-level attention."""

    def __init__(self, in_dim, num_views=2, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.num_views = num_views
        self.semantic_attention = nn.Parameter(torch.empty(in_dim, 1))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.semantic_attention)

    def forward(self, views):
        if len(views) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} semantic views, got {len(views)}"
            )
        stacked = torch.stack(tuple(views), dim=1)
        scores = torch.matmul(stacked, self.semantic_attention).squeeze(-1)
        weights = self.dropout(torch.softmax(scores, dim=1)).unsqueeze(-1)
        return torch.sum(stacked * weights, dim=1)


# Keep the name used by the older compact implementation available to callers
# that imported it directly; the RQ2 model uses the clearer SemanticFusion name.
SemanticLevelAttention = SemanticFusion


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


class FeedForwardEncoder(nn.Module):
    """Node-wise encoder with no graph access and near-matched capacity."""

    def __init__(self, dimension, target_parameters, dropout):
        super().__init__()
        # Linear(d,h) + Linear(h,d) + LayerNorm(d)
        coefficient = 2 * dimension + 1
        fixed = 3 * dimension
        hidden = max(1, round((target_parameters - fixed) / coefficient))
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.LayerNorm(dimension),
        )
        self.target_parameters = target_parameters

    def forward(self, _, features):
        return self.network(features)


def _reference_encoder(args, graph, num_entities):
    encoder = HANEncoder if args.use_han else GATEncoder
    if args.use_han:
        return encoder(graph, num_entities, args.dimension, args.heads, args.dropout)
    return encoder(args.dimension, args.heads, args.dropout)


def _context_encoding_enabled(args):
    """Resolve the paper switch while preserving the legacy low-level flag."""
    setting = getattr(args, "use_context_encoding", None)
    if setting is None:
        return bool(getattr(args, "use_graph_encoder", True))
    return bool(setting)


class FedGLModel(nn.Module):
    def __init__(
        self,
        args,
        user_graph,
        service_graph,
        num_users,
        num_services,
        local_interactions=None,
    ):
        super().__init__()
        self.args = args
        self.user_graph = user_graph
        self.service_graph = service_graph
        self.num_users = num_users
        self.num_services = num_services
        dimension = args.dimension
        self.local_interactions = tuple(
            sorted({(int(user), int(service)) for user, service in (local_interactions or [])})
        )
        for user_id, service_id in self.local_interactions:
            if not 0 <= user_id < self.num_users:
                raise ValueError(f"Local interaction contains invalid user ID {user_id}")
            if not 0 <= service_id < self.num_services:
                raise ValueError(
                    f"Local interaction contains invalid service ID {service_id}"
                )

        self.use_context_encoding = _context_encoding_enabled(args)
        # Interaction fusion is a separate RQ3 factor.  It is meaningful only
        # after context encoding; direct embeddings must never instantiate or
        # invoke the private interaction path even if a stale low-level flag
        # requests it.
        self.use_interaction_fusion = bool(
            getattr(args, "use_interaction_fusion", True)
        ) and self.use_context_encoding
        # A caller using only the historical --use_graph_encoder=false switch
        # still receives its old feed-forward replacement.  Named RQ2 presets
        # set use_context_encoding explicitly and therefore use strict direct
        # embeddings when that switch is false.
        self._legacy_feedforward_encoder = (
            getattr(args, "use_context_encoding", None) is None
            and not bool(getattr(args, "use_graph_encoder", True))
        )
        self.use_graph_encoder = self.use_context_encoding

        self.user_embeds = nn.Embedding(user_graph.num_nodes(), dimension)
        self.service_embeds = nn.Embedding(service_graph.num_nodes(), dimension)
        nn.init.xavier_uniform_(self.user_embeds.weight)
        nn.init.xavier_uniform_(self.service_embeds.weight)

        self.user_encoder = None
        self.service_encoder = None
        if self.use_context_encoding:
            self.user_encoder = _reference_encoder(args, user_graph, num_users)
            self.service_encoder = _reference_encoder(args, service_graph, num_services)
            if self.use_interaction_fusion:
                interaction_dropout = getattr(args, "dropout", 0.1)
                self.interaction_attention = BipartiteInteractionAttention(
                    dimension,
                    dimension,
                    dropout=interaction_dropout,
                )
                self.interaction_layer_norm = nn.LayerNorm(dimension)
                self.user_semantic_fusion = SemanticFusion(
                    dimension, num_views=2, dropout=interaction_dropout
                )
                self.service_semantic_fusion = SemanticFusion(
                    dimension, num_views=2, dropout=interaction_dropout
                )
        elif self._legacy_feedforward_encoder:
            # Measure the enabled encoder without consuming the replacement's
            # initialization stream, then create graph-free node-wise encoders.
            rng_state = torch.get_rng_state()
            reference = _reference_encoder(args, user_graph, num_users)
            target_parameters = _parameter_count(reference)
            del reference
            torch.set_rng_state(rng_state)
            self.user_encoder = FeedForwardEncoder(
                dimension, target_parameters, args.dropout
            )
            self.service_encoder = FeedForwardEncoder(
                dimension, target_parameters, args.dropout
            )

        input_dim = 2 * dimension
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(args.dropout)

        if args.task_mode == "single":
            hidden = [128, 64] if args.use_asymmetric_head else [64]
            self.sr_head = _mlp(input_dim, hidden, 1)
        elif args.use_asymmetric_head:
            self.sr_head = _mlp(input_dim, [128, 64], 1)
            self.aux_head = _mlp(input_dim, [32], 3)
        else:
            # RQ2's symmetric variant gives both the SR branch and the
            # auxiliary (RB/RH/RTT) branch the same [128, 64] hidden widths.
            # The output remains one SR value plus three auxiliary values.
            self.sr_head = _mlp(input_dim, [128, 64], 1)
            self.aux_head = _mlp(input_dim, [128, 64], 3)
        self._interaction_graph_cache = None
        self._interaction_graph_device = None
        self._cache = None

    def clear_cache(self):
        self._cache = None

    def encode(self):
        user = self.user_embeds.weight
        service = self.service_embeds.weight
        if self.use_context_encoding:
            user = self.user_encoder(self.user_graph, user)
            service = self.service_encoder(self.service_graph, service)
            user_entities = user[: self.num_users]
            service_entities = service[: self.num_services]
            if self.use_interaction_fusion:
                interaction_graph = self._get_local_interaction_graph(user.device)
                interaction = self.interaction_attention(
                    interaction_graph, user_entities, service_entities
                )
                interaction = self.interaction_layer_norm(F.elu(interaction))
                user = self.user_semantic_fusion(
                    [user_entities, interaction[: self.num_users]]
                )
                service = self.service_semantic_fusion(
                    [service_entities, interaction[self.num_users :]]
                )
            else:
                # RQ3 w/o Interaction: retain the context encoder output as
                # the entity representation and feed it directly to the head.
                user = user_entities
                service = service_entities
        elif self._legacy_feedforward_encoder:
            user = self.user_encoder(self.user_graph, user)
            service = self.service_encoder(self.service_graph, service)
        return user[: self.num_users], service[: self.num_services]

    def _get_local_interaction_graph(self, device):
        """Build/cache a bidirectional graph from training-only local calls."""
        device = torch.device(device)
        if (
            self._interaction_graph_cache is not None
            and self._interaction_graph_device == device
        ):
            return self._interaction_graph_cache

        num_nodes = self.num_users + self.num_services
        self_loops = torch.arange(num_nodes, dtype=torch.int64, device=device)
        if self.local_interactions:
            user_ids = torch.tensor(
                [pair[0] for pair in self.local_interactions],
                dtype=torch.int64,
                device=device,
            )
            service_ids = torch.tensor(
                [pair[1] + self.num_users for pair in self.local_interactions],
                dtype=torch.int64,
                device=device,
            )
            source = torch.cat((user_ids, service_ids, self_loops))
            target = torch.cat((service_ids, user_ids, self_loops))
        else:
            source = self_loops
            target = self_loops
        self._interaction_graph_cache = dgl.graph(
            (source, target), num_nodes=num_nodes, device=device
        )
        self._interaction_graph_device = device
        return self._interaction_graph_cache

    def prepare_eval(self):
        with torch.no_grad():
            self._cache = self.encode()

    def forward(self, user_index, service_index, use_cache=False):
        user, service = self._cache if use_cache and self._cache is not None else self.encode()
        features = torch.cat((user[user_index], service[service_index]), dim=1)
        features = self.input_dropout(self.input_norm(features))

        if self.args.task_mode == "single":
            output = self.sr_head(features).squeeze(1)
        else:
            output = torch.cat((self.sr_head(features), self.aux_head(features)), dim=1)
        return torch.sigmoid(output)
