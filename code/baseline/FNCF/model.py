"""Multi-dimensional context-aware neural collaborative filtering."""

import torch
from torch import nn


def _context_incidence(graph, num_users):
    """Return normalized user-to-context incidence from the private DGL graph."""
    context_count = max(int(graph.num_nodes()) - int(num_users), 1)
    incidence = torch.zeros(
        int(num_users),
        context_count,
        dtype=torch.float32,
        device=graph.device,
    )
    source, target = graph.edges()
    mask = (source < num_users) & (target >= num_users)
    if bool(mask.any()):
        user = source[mask].long()
        context = (target[mask] - num_users).long()
        incidence[user, context] = 1.0
    return incidence / incidence.sum(dim=1, keepdim=True).clamp_min(1.0)


class FNCFModel(nn.Module):
    """User context + user ID + peer ID embeddings followed by an MLP."""

    def __init__(self, args, user_graph, num_users, num_services):
        super().__init__()
        dimension = int(args.dimension)
        context_count = max(int(user_graph.num_nodes()) - int(num_users), 1)
        output_dim = 1 if args.task_mode == "single" else 4

        self.user_embedding = nn.Embedding(num_users, dimension)
        self.peer_embedding = nn.Embedding(num_services, dimension)
        self.context_embedding = nn.Embedding(context_count, dimension)
        for embedding in (
            self.user_embedding,
            self.peer_embedding,
            self.context_embedding,
        ):
            nn.init.xavier_uniform_(embedding.weight)

        # The incidence is client-private graph data, not a federated parameter.
        self.register_buffer(
            "context_incidence",
            _context_incidence(user_graph, num_users),
            persistent=False,
        )

        layers = []
        width = 3 * dimension
        for hidden in args.fncf_hidden_dims:
            layers.extend(
                [
                    nn.Linear(width, int(hidden)),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                ]
            )
            width = int(hidden)
        layers.append(nn.Linear(width, output_dim))
        self.predictor = nn.Sequential(*layers)

    def clear_cache(self):
        return None

    def prepare_eval(self):
        return None

    def forward(self, user_index, service_index, use_cache=False):
        del use_cache
        context_table = self.context_incidence @ self.context_embedding.weight
        features = torch.cat(
            (
                context_table[user_index],
                self.user_embedding(user_index),
                self.peer_embedding(service_index),
            ),
            dim=1,
        )
        prediction = torch.sigmoid(self.predictor(features))
        return prediction.squeeze(1) if prediction.shape[1] == 1 else prediction
