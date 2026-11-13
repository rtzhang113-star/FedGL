"""Federated matrix-factorization model used by FL-MFGM."""

import math

import torch
from torch import nn


class FLMFGMModel(nn.Module):
    """Task-wise user/peer factorization adapted to the FedGL data protocol.

    The paper predicts one SuccessRate matrix.  The active experiment protocol
    contains four normalized reliability matrices, so multi-task mode gives
    every task an independent factor slice while retaining the paper's dot
    product predictor.
    """

    def __init__(self, args, num_users, num_services):
        super().__init__()
        self.task_count = 1 if args.task_mode == "single" else 4
        self.dimension = int(args.dimension)
        width = self.task_count * self.dimension
        self.user_factors = nn.Embedding(num_users, width)
        self.peer_factors = nn.Embedding(num_services, width)
        nn.init.normal_(self.user_factors.weight, std=1.0 / math.sqrt(self.dimension))
        nn.init.normal_(self.peer_factors.weight, std=1.0 / math.sqrt(self.dimension))

    def clear_cache(self):
        return None

    def prepare_eval(self):
        return None

    def forward(self, user_index, service_index, use_cache=False):
        del use_cache
        user = self.user_factors(user_index).reshape(
            -1, self.task_count, self.dimension
        )
        peer = self.peer_factors(service_index).reshape(
            -1, self.task_count, self.dimension
        )
        prediction = torch.sigmoid((user * peer).sum(dim=-1))
        return prediction.squeeze(1) if self.task_count == 1 else prediction
