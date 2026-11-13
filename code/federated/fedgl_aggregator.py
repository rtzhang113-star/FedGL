"""FedGL core: learned client graphs and task-specific layer routing.

The historical implementation always allocated four task graphs.  RQ3's
Single Task condition is intentionally narrower: only the SR graph and its
associated parameter routing are active.  The module keeps ``TASKS`` as the
four-task compatibility constant for metrics, while all graph operations use
the active task count derived from the run configuration.
"""

import copy
import itertools
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from core_utils.common_utils import clone_state
from federated.aggregator import cosine_graph, weighted_average


TASKS = ("SR", "RB", "RH", "RTT")


def _active_task_count(args):
    """Return the number of task graphs participating in this run."""
    configured = getattr(args, "active_task_count", None)
    if configured is not None:
        return int(configured)
    return 1 if getattr(args, "task_mode", "multi") == "single" else len(TASKS)


class TaskGraphLearner(nn.Module):
    """Directed query-key graph generator with optional prior message passing."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.active_task_count = _active_task_count(args)
        count = 1 if args.task_graph_mode == "shared" else self.active_task_count
        feature_dim = args.client_feature_dim
        hidden_dim = args.meta_graph_hidden_dim
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(count)
            ]
        )
        self.queries = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(count)]
        )
        self.keys = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(count)]
        )
        self.gate_scale = nn.Parameter(torch.ones(count))
        self.gate_bias = nn.Parameter(torch.zeros(count))

    @staticmethod
    def _row_softmax(scores):
        if scores.shape[0] > 1:
            scores = scores.masked_fill(
                torch.eye(scores.shape[0], device=scores.device, dtype=torch.bool),
                -1e9,
            )
        return torch.softmax(scores, dim=1)

    def forward(self, features, prior=None):
        shared = self.args.task_graph_mode == "shared"
        inputs = [features.mean(0)] if shared else list(features)
        priors = [prior.mean(0)] if shared and prior is not None else prior
        dynamic_graphs, fused_graphs = [], []

        for task_id, task_features in enumerate(inputs):
            task_prior = None if prior is None else priors[task_id]
            if self.args.graph_generation_method == "gnn":
                if task_prior is None:
                    seed = self._row_softmax(
                        F.normalize(task_features, dim=1)
                        @ F.normalize(task_features, dim=1).T
                    )
                else:
                    seed = task_prior
                task_features = task_features + seed @ task_features

            hidden = self.encoders[task_id](task_features)
            scores = self.queries[task_id](hidden) @ self.keys[task_id](hidden).T
            scores = scores / math.sqrt(hidden.shape[1])
            dynamic = self._row_softmax(
                scores / max(self.args.attention_temperature, 1e-6)
            )

            if task_prior is None:
                fused = dynamic
            else:
                gate = torch.sigmoid(
                    self.gate_scale[task_id] * dynamic * task_prior
                    + self.gate_bias[task_id]
                )
                rho = self.args.graph_prior_weight
                fused = (1 - rho) * dynamic + rho * (
                    gate * dynamic + (1 - gate) * task_prior
                )
                fused = fused / fused.sum(1, keepdim=True).clamp_min(1e-12)
            dynamic_graphs.append(dynamic)
            fused_graphs.append(fused)

        dynamic = torch.stack(dynamic_graphs)
        fused = torch.stack(fused_graphs)
        if shared and self.active_task_count > 1:
            dynamic = dynamic.repeat(self.active_task_count, 1, 1)
            fused = fused.repeat(self.active_task_count, 1, 1)
        return dynamic, fused


class FedGLAggregator:
    def __init__(self, args, prior=None):
        self.args = args
        self.device = torch.device(args.devices)
        self.active_task_count = _active_task_count(args)
        self.active_tasks = TASKS[: self.active_task_count]
        self.prior = self._active_prior(prior)
        self.learner = TaskGraphLearner(args).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.learner.parameters(), lr=args.task_graph_lr
        )
        self.previous_graphs = None
        self.previous_clients = None
        # Kept separately from ``previous_graphs`` because the latter is used
        # by the graph-learning smoothness loss.  These tensors represent the
        # previous communication round solely for D_round diagnostics.
        self.diagnostic_previous_graphs = None
        self.diagnostic_previous_clients = None
        self.last_graph_update_time = float("nan")
        self.static_graphs = (
            self._initialize_static_graphs() if args.freeze_task_graphs else None
        )

    def _active_prior(self, prior):
        """Normalize an optional prior to the active task cardinality.

        ``build_client_prior`` historically emits four identical task slices.
        Single Task consumes only the first (SR) slice and retains a leading
        task dimension so all graph-learning code can stay tensor-shaped.
        """
        if prior is None:
            return None
        prior = prior.to(self.device)
        if prior.ndim == 2:
            prior = prior.unsqueeze(0)
        if prior.ndim != 3:
            raise ValueError("prior must have shape [tasks, clients, clients]")
        if prior.shape[0] < self.active_task_count:
            raise ValueError(
                "prior has fewer task slices than the active task count"
            )
        return prior[: self.active_task_count].clone()

    def _initialize_static_graphs(self):
        """Create fixed full-client graphs before federated training."""
        count = self.args.clients
        if self.prior is None:
            base = torch.eye(count, device=self.device).unsqueeze(0).repeat(
                self.active_task_count, 1, 1
            )
            prior = None
        else:
            base = self.prior.to(self.device)
            prior = base
        features = torch.stack(
            [
                torch.stack([self._compress(row) for row in task_graph])
                for task_graph in base
            ]
        )
        with torch.no_grad():
            _, graphs = self.learner(features, prior)
        return graphs.detach()

    def _selected_static_graphs(self, client_ids):
        ids = torch.tensor(client_ids, device=self.device)
        graphs = self.static_graphs.index_select(1, ids).index_select(2, ids)
        row_sums = graphs.sum(2, keepdim=True)
        graphs = graphs / row_sums.clamp_min(1e-12)
        if torch.any(row_sums <= 1e-12):
            identity = torch.eye(len(client_ids), device=self.device)[None]
            graphs = torch.where(row_sums <= 1e-12, identity, graphs)
        return graphs

    def _selected_prior(self, client_ids):
        if self.prior is None or self.args.graph_prior_weight == 0:
            return None
        ids = torch.tensor(client_ids, device=self.prior.device)
        prior = self.prior.index_select(1, ids).index_select(2, ids)
        return prior / prior.sum(2, keepdim=True).clamp_min(1e-12)

    def _compress(self, vector):
        size = self.args.client_feature_dim
        if vector.numel() > size:
            return F.adaptive_avg_pool1d(vector[None, None], size).flatten()
        return F.pad(vector, (0, size - vector.numel()))

    def _sync_device(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _task_update(self, state, reference, task_id):
        chunks = []
        shared_representation_prefixes = (
            "user_embeds.",
            "service_embeds.",
            "user_encoder.",
            "service_encoder.",
            "interaction_attention.",
            "interaction_layer_norm.",
            "user_semantic_fusion.",
            "service_semantic_fusion.",
            "input_norm.",
        )
        for key, value in state.items():
            if not value.is_floating_point():
                continue
            delta = value.float() - reference[key].float()
            shared = key.startswith(shared_representation_prefixes)
            relevant = shared

            if key.startswith("sr_head"):
                relevant = task_id == 0
            elif key.startswith("aux_head"):
                relevant = task_id > 0
                if relevant and delta.ndim and delta.shape[0] == 3:
                    delta = delta[task_id - 1]
            elif key.startswith("symmetric_head"):
                parts = key.split(".")
                head_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                relevant = head_id is None or head_id == task_id
                if head_id is None and delta.ndim and delta.shape[0] == 4:
                    delta = delta[task_id]
            if relevant:
                chunks.append(delta.flatten())

        if not chunks:
            chunks = [
                (value.float() - reference[key].float()).flatten()
                for key, value in state.items()
                if value.is_floating_point()
            ]
        return self._compress(torch.cat(chunks))

    def _features(self, states, reference):
        return torch.stack(
            [
                torch.stack(
                    [self._task_update(state, reference, task) for state in states]
                )
                for task in range(self.active_task_count)
            ]
        ).to(self.device)

    def _learn_graphs(self, states, client_ids, reference):
        self._sync_device()
        graph_start = time.perf_counter()
        if self.static_graphs is not None:
            result = (
                self._selected_static_graphs(client_ids),
                self._selected_prior(client_ids),
                0.0,
            )
            self._sync_device()
            self.last_graph_update_time = time.perf_counter() - graph_start
            return result

        features = self._features(states, reference).detach()
        similarity = F.normalize(features, dim=2) @ F.normalize(
            features, dim=2
        ).transpose(1, 2)
        distance = (1 - similarity).clamp_min(0)
        prior = self._selected_prior(client_ids)
        current_common = previous_common = None
        if self.previous_graphs is not None:
            common = sorted(set(client_ids) & set(self.previous_clients))
            if len(common) >= 2:
                current_pos = torch.tensor(
                    [client_ids.index(client) for client in common],
                    device=self.device,
                )
                previous_pos = torch.tensor(
                    [self.previous_clients.index(client) for client in common],
                    device=self.device,
                )
                previous_common = self.previous_graphs.index_select(
                    1, previous_pos
                ).index_select(2, previous_pos)
        loss_value = 0.0

        for _ in range(self.args.graph_evolution_epochs):
            dynamic, fused = self.learner(features, prior)
            graph_for_loss = fused
            if fused.shape[1] > 1:
                diagonal = torch.eye(
                    fused.shape[1], device=fused.device, dtype=torch.bool
                )[None]
                graph_for_loss = fused.masked_fill(diagonal, 0)
                graph_for_loss = graph_for_loss / graph_for_loss.sum(
                    2, keepdim=True
                ).clamp_min(1e-12)
            structure = (graph_for_loss * distance).sum(2).mean()
            entropy = -(
                graph_for_loss.clamp_min(1e-12).log() * graph_for_loss
            ).sum(2).mean()
            if previous_common is None:
                smoothness = fused.new_zeros(())
            else:
                current_common = fused.index_select(
                    1, current_pos
                ).index_select(2, current_pos)
                current_common = current_common / current_common.sum(
                    2, keepdim=True
                ).clamp_min(1e-12)
                old_common = previous_common / previous_common.sum(
                    2, keepdim=True
                ).clamp_min(1e-12)
                smoothness = F.mse_loss(current_common, old_common)
            loss = (
                structure
                + self.args.graph_sparsity * entropy
                + self.args.graph_smoothness * smoothness
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach())

        with torch.no_grad():
            _, graphs = self.learner(features, prior)
        self.previous_graphs = graphs.detach()
        self.previous_clients = tuple(client_ids)
        self._sync_device()
        self.last_graph_update_time = time.perf_counter() - graph_start
        return graphs.detach(), prior, loss_value

    def _diagnostic_round_distance(self, graphs, client_ids):
        """Compare aligned current/previous task graphs for D_round.

        The graph tensors may cover different client subsets between rounds.
        We therefore align by client id, extract the corresponding submatrices,
        and re-normalize every row before computing the task-wise Frobenius
        distances.  A first round or fewer than two common clients has no
        defined distance.
        """
        previous_graphs = self.diagnostic_previous_graphs
        previous_clients = self.diagnostic_previous_clients
        if (
            previous_graphs is None
            or previous_clients is None
            or client_ids is None
        ):
            return math.nan

        common = sorted(set(client_ids) & set(previous_clients))
        if len(common) < 2:
            return math.nan
        if graphs.shape[0] != previous_graphs.shape[0]:
            return math.nan

        current_pos = torch.tensor(
            [client_ids.index(client) for client in common],
            device=graphs.device,
        )
        previous_pos = torch.tensor(
            [previous_clients.index(client) for client in common],
            device=graphs.device,
        )
        old = previous_graphs.to(graphs.device).index_select(1, previous_pos)
        old = old.index_select(2, previous_pos)
        current = graphs.index_select(1, current_pos).index_select(2, current_pos)
        current = current / current.sum(2, keepdim=True).clamp_min(1e-12)
        old = old / old.sum(2, keepdim=True).clamp_min(1e-12)
        task_count = graphs.shape[0]
        distance = sum(
            torch.linalg.norm(current[task] - old[task])
            for task in range(task_count)
        )
        return float(distance.detach()) / (task_count * len(common))

    def _diagnostics(self, graphs, graph_loss, client_ids=None):
        pairs = list(itertools.combinations(range(self.active_task_count), 2))
        if not pairs:
            return {}
        n = graphs.shape[1]
        k = min(self.args.task_graph_topk, max(n - 1, 1))
        jaccard, frobenius, spectral = [], [], []

        spectra = []
        for graph in graphs:
            symmetric = 0.5 * (graph + graph.T)
            degree = symmetric.sum(1).clamp_min(1e-12)
            inv = degree.rsqrt()
            laplacian = torch.eye(n, device=graph.device) - (
                inv[:, None] * symmetric * inv[None, :]
            )
            spectra.append(torch.linalg.eigvalsh(laplacian))

        neighbour_scores = graphs.clone()
        if n > 1:
            neighbour_scores = neighbour_scores.masked_fill(
                torch.eye(n, device=graphs.device, dtype=torch.bool)[None],
                -1.0,
            )
        neighbours = neighbour_scores.topk(k, dim=2).indices
        for left, right in pairs:
            task_overlap = []
            for client in range(n):
                a = set(neighbours[left, client].tolist())
                b = set(neighbours[right, client].tolist())
                task_overlap.append(len(a & b) / max(len(a | b), 1))
            jaccard.append(sum(task_overlap) / len(task_overlap))
            frobenius.append(
                float(torch.linalg.norm(graphs[left] - graphs[right]) / n)
            )
            spectral.append(
                float(torch.linalg.norm(spectra[left] - spectra[right]) / n)
            )
        d_task = sum(frobenius) / len(frobenius)
        diagnostics = {
            "graph_loss": graph_loss,
            "graph_update_time": self.last_graph_update_time,
            "topk_jaccard": sum(jaccard) / len(jaccard),
            "frobenius_distance": sum(frobenius) / len(frobenius),
            "spectral_distance": sum(spectral) / len(spectral),
            "d_task": d_task,
            "d_round": self._diagnostic_round_distance(graphs, client_ids),
        }
        # Update only the diagnostic state after measuring the current round;
        # the graph-learning ``previous_graphs`` state above remains untouched.
        if client_ids is not None:
            self.diagnostic_previous_graphs = graphs.detach().clone()
            self.diagnostic_previous_clients = tuple(client_ids)
        return diagnostics

    @staticmethod
    def _mix(stack, graph):
        return torch.tensordot(graph, stack, dims=([1], [0]))

    def _collaborative_states(self, states, graphs, prior, sample_sizes, reference):
        mode = self.args.layer_routing_mode
        count = len(states)
        identity = torch.eye(count, device=self.device)
        data_weight = torch.tensor(sample_sizes, device=self.device, dtype=torch.float32)
        data_weight = data_weight / data_weight.sum().clamp_min(1)
        global_graph = data_weight[None].repeat(count, 1)
        attention_graph = cosine_graph(
            states, reference, self.args.attention_temperature
        ).to(self.device)
        mean_graph = graphs.mean(0)
        prior_graph = None if prior is None else prior.mean(0)
        personalized = [dict() for _ in states]

        for key in states[0]:
            values = [state[key] for state in states]
            if not values[0].is_floating_point():
                for index, value in enumerate(values):
                    personalized[index][key] = value.clone()
                continue

            stack = torch.stack([value.float() for value in values]).to(self.device)
            row_graphs = None
            if mode == "local":
                graph = identity
            elif mode == "uniform_conv":
                graph = mean_graph
            elif mode == "uniform_att":
                graph = attention_graph
            elif mode == "heuristic":
                graph = identity if "embeds" in key else global_graph
            elif "embeds" in key:
                # In Single Task the only valid routing topology is the
                # learned SR graph itself.  Multi-task runs retain the
                # established context-prior routing for shared embeddings.
                if self.active_task_count == 1:
                    graph = graphs[0]
                else:
                    graph = prior_graph if prior_graph is not None else mean_graph
            elif key.startswith("sr_head"):
                graph = graphs[0]
            elif (
                self.active_task_count == len(TASKS)
                and key.startswith("aux_head")
                and stack.ndim > 1
                and stack.shape[1] == 3
            ):
                row_graphs = [graphs[1], graphs[2], graphs[3]]
                graph = None
            elif self.active_task_count == len(TASKS) and key.startswith("aux_head"):
                graph = graphs[1:].mean(0)
            elif key.startswith("symmetric_head."):
                parts = key.split(".")
                head_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                graph = (
                    mean_graph
                    if head_id is None or head_id >= self.active_task_count
                    else graphs[head_id]
                )
            elif (
                self.active_task_count == len(TASKS)
                and key.startswith("symmetric_head")
                and stack.ndim > 1
                and stack.shape[1] == 4
            ):
                row_graphs = list(graphs)
                graph = None
            else:
                graph = mean_graph

            if row_graphs is None:
                collaborative = self._mix(stack, graph)
            else:
                collaborative = torch.empty_like(stack)
                for row, row_graph in enumerate(row_graphs):
                    collaborative[:, row] = self._mix(stack[:, row], row_graph)

            beta = 0.0 if mode == "local" else self.args.serverbeta
            mixed = (1 - beta) * stack + beta * collaborative
            for index, value in enumerate(values):
                personalized[index][key] = mixed[index].cpu().to(value.dtype)
        return personalized

    def aggregate(self, states, client_ids, sample_sizes, reference):
        global_reference = weighted_average(states, sample_sizes)
        graphs, prior, graph_loss = self._learn_graphs(
            states, client_ids, reference
        )
        personalized = self._collaborative_states(
            states, graphs, prior, sample_sizes, reference
        )
        return (
            global_reference,
            {
                client_id: state
                for client_id, state in zip(client_ids, personalized)
            },
            self._diagnostics(graphs, graph_loss, client_ids),
        )

    def snapshot(self):
        return {
            "learner": clone_state(self.learner.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "previous_graphs": (
                None
                if self.previous_graphs is None
                else self.previous_graphs.detach().cpu().clone()
            ),
            "previous_clients": self.previous_clients,
            "diagnostic_previous_graphs": (
                None
                if self.diagnostic_previous_graphs is None
                else self.diagnostic_previous_graphs.detach().cpu().clone()
            ),
            "diagnostic_previous_clients": self.diagnostic_previous_clients,
            "static_graphs": (
                None
                if self.static_graphs is None
                else self.static_graphs.detach().cpu().clone()
            ),
            "last_graph_update_time": self.last_graph_update_time,
        }

    def restore(self, state):
        self.learner.load_state_dict(state["learner"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.previous_graphs = (
            None
            if state.get("previous_graphs") is None
            else state["previous_graphs"].to(self.device)
        )
        self.previous_clients = state.get("previous_clients")
        self.diagnostic_previous_graphs = (
            None
            if state.get("diagnostic_previous_graphs") is None
            else state["diagnostic_previous_graphs"].to(self.device)
        )
        self.diagnostic_previous_clients = state.get(
            "diagnostic_previous_clients"
        )
        if state.get("static_graphs") is not None:
            self.static_graphs = state["static_graphs"].to(self.device)
        self.last_graph_update_time = state.get(
            "last_graph_update_time", float("nan")
        )
