"""Minimal federated training loop for paper experiments."""

import time
from pathlib import Path

import numpy as np
import torch

from baseline import build_specialized_aggregator
from core_utils.common_utils import (
    TASKS,
    clone_state,
    combine_client_metrics,
    state_bytes,
)
from federated.aggregator import (
    PERSONALIZED_METHODS,
    BaselineAggregator,
)
from federated.fedgl_aggregator import FedGLAggregator


class FederatedServer:
    def __init__(self, args, clients, prior=None):
        self.args = args
        self.clients = clients
        self.device = torch.device(args.devices)
        self.personalized = args.agg in PERSONALIZED_METHODS
        self.global_state = clone_state(clients[0].model.state_dict())
        self.client_states = {
            client.client_id: clone_state(self.global_state) for client in clients
        }
        specialized = build_specialized_aggregator(args, prior)
        self.aggregator = (
            FedGLAggregator(args, prior)
            if args.agg == "fedgl"
            else specialized or BaselineAggregator(args, prior)
        )
        self.history = self._empty_history()

    @staticmethod
    def _empty_history():
        history = {
            "round": [],
            "local_time": [],
            "local_time_mean": [],
            "aggregation_time": [],
            "communication_mb": [],
            "downlink_mb": [],
            "uplink_mb": [],
            "cumulative_communication_mb": [],
            "participating_clients": [],
            "round_time": [],
            "graph_update_time": [],
            "peak_server_vram_mb": [],
            "peak_server_reserved_vram_mb": [],
            "is_warmup_round": [],
            "is_timed_round": [],
            "graph_loss": [],
            "topk_jaccard": [],
            "frobenius_distance": [],
            "spectral_distance": [],
            "d_task": [],
            "d_round": [],
        }
        for split in ("train", "valid"):
            for metric in ("loss", "MAE", "RMSE"):
                history[f"{split}_{metric.lower()}"] = []
            for task in TASKS:
                history[f"{split}_{task.lower()}_mae"] = []
                history[f"{split}_{task.lower()}_rmse"] = []
        return history

    def _sync_device(self):
        """Synchronize only for CUDA timing boundaries."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _cuda_peak_memory(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return float("nan"), float("nan")
        return (
            torch.cuda.max_memory_allocated(self.device) / 1024**2,
            torch.cuda.max_memory_reserved(self.device) / 1024**2,
        )

    def _state_for(self, client_id):
        state = (
            self.client_states[client_id]
            if self.personalized
            else self.global_state
        )
        if self.personalized and hasattr(
            self.aggregator, "prepare_client_state"
        ):
            return self.aggregator.prepare_client_state(
                state, self.global_state
            )
        return state

    def _select_clients(self, round_index):
        active = [
            client.client_id for client in self.clients if client.train_size > 0
        ]
        if not active:
            raise RuntimeError("no client has local training samples")
        count = min(
            len(active), max(1, int(self.args.clients * self.args.client_frac))
        )
        rng = np.random.default_rng(self.args.random_seed + round_index)
        return sorted(rng.choice(active, count, replace=False).tolist())

    def _reference(self):
        return (
            self.global_state
            if self.args.agg in {"fedprox", "fedgl"}
            else None
        )

    def run_round(self, round_index):
        self._sync_device()
        round_start = time.perf_counter()
        client_ids = self._select_clients(round_index)
        states, metrics, sample_sizes = [], [], []
        local_times, downlink_bytes, uploaded = [], 0, 0
        shared_reference = self._reference()

        for client_id in client_ids:
            start_state = self._state_for(client_id)
            reference = (
                start_state
                if self.args.agg == "pfedgraph"
                else shared_reference
            )
            downlink_bytes += (
                self.aggregator.downlink_bytes(start_state)
                if hasattr(self.aggregator, "downlink_bytes")
                else state_bytes(start_state)
            )
            if (
                reference is not None
                and self.personalized
                and start_state is not reference
            ):
                downlink_bytes += state_bytes(reference)
            client = self.clients[client_id]
            client.load_state(start_state, reference)
            self._sync_device()
            local_start = time.perf_counter()
            state, client_metrics, elapsed = client.train()
            self._sync_device()
            measured_elapsed = time.perf_counter() - local_start
            states.append(state)
            metrics.append(client_metrics)
            sample_sizes.append(client.train_size)
            # ``elapsed`` is retained by the client for legacy callers, but
            # RQ5 uses the synchronized per-client boundary measured here.
            del elapsed
            local_times.append(measured_elapsed)
            upload_size = state_bytes(state)
            uploaded += upload_size

        self._sync_device()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        aggregation_start = time.perf_counter()
        if self.args.agg == "fedgl":
            global_state, personalized, diagnostics = self.aggregator.aggregate(
                states, client_ids, sample_sizes, self.global_state
            )
        else:
            global_state, personalized = self.aggregator.aggregate(
                states, client_ids, sample_sizes, self.global_state
            )
            diagnostics = {}
        if hasattr(self.aggregator, "last_upload_bytes"):
            uploaded = int(self.aggregator.last_upload_bytes)
        self.global_state = clone_state(global_state)
        if self.personalized:
            self.client_states.update(
                {client_id: clone_state(state) for client_id, state in personalized.items()}
            )
        self._sync_device()
        aggregation_time = time.perf_counter() - aggregation_start
        peak_allocated, peak_reserved = self._cuda_peak_memory()
        local_time = sum(local_times)
        local_time_mean = (
            local_time / len(local_times) if local_times else float("nan")
        )
        downlink_mb = downlink_bytes / 1024**2
        uplink_mb = uploaded / 1024**2
        communication_mb = downlink_mb + uplink_mb
        round_time = time.perf_counter() - round_start
        return (
            combine_client_metrics(metrics),
            diagnostics,
            local_time,
            aggregation_time,
            communication_mb,
            len(client_ids),
            local_time_mean,
            downlink_mb,
            uplink_mb,
            round_time,
            diagnostics.get("graph_update_time", float("nan")),
            peak_allocated,
            peak_reserved,
        )

    def evaluate(self, split):
        metrics = []
        client_rmse = []
        client_eval_count = 0
        for client in self.clients:
            client.load_state(self._state_for(client.client_id))
            client_metrics = client.evaluate(split)
            metrics.append(client_metrics)
            if client_metrics.get("n", 0) > 0:
                client_eval_count += 1
                value = client_metrics.get("RMSE", float("nan"))
                if np.isfinite(value):
                    client_rmse.append(float(value))

        combined = combine_client_metrics(metrics)
        if client_rmse:
            combined["Client_Macro_RMSE_Median"] = float(np.median(client_rmse))
            combined["Client_Macro_RMSE_P90"] = float(
                np.percentile(client_rmse, 90)
            )
        else:
            combined["Client_Macro_RMSE_Median"] = float("nan")
            combined["Client_Macro_RMSE_P90"] = float("nan")
        combined["Client_Eval_Count"] = client_eval_count
        return combined

    def _record_metrics(self, split, metrics):
        for metric in ("loss", "MAE", "RMSE"):
            self.history[f"{split}_{metric.lower()}"].append(
                metrics.get(metric, float("nan"))
            )
        for task in TASKS:
            self.history[f"{split}_{task.lower()}_mae"].append(
                metrics.get(f"{task}_MAE", float("nan"))
            )
            self.history[f"{split}_{task.lower()}_rmse"].append(
                metrics.get(f"{task}_RMSE", float("nan"))
            )

    def _record_round(
        self,
        round_index,
        train_metrics,
        valid_metrics,
        diagnostics,
        local_time,
        aggregation_time,
        communication_mb,
        participating_clients=None,
        local_time_mean=None,
        downlink_mb=None,
        uplink_mb=None,
        round_time=None,
        graph_update_time=None,
        peak_server_vram_mb=None,
        peak_server_reserved_vram_mb=None,
    ):
        self.history["round"].append(round_index)
        self.history["local_time"].append(local_time)
        self.history["local_time_mean"].append(
            float("nan") if local_time_mean is None else local_time_mean
        )
        self.history["aggregation_time"].append(aggregation_time)
        self.history["communication_mb"].append(communication_mb)
        self.history["downlink_mb"].append(
            float("nan") if downlink_mb is None else downlink_mb
        )
        self.history["uplink_mb"].append(
            float("nan") if uplink_mb is None else uplink_mb
        )
        self.history["participating_clients"].append(
            len(self._select_clients(round_index))
            if participating_clients is None
            else participating_clients
        )
        self.history["round_time"].append(
            float("nan") if round_time is None else round_time
        )
        self.history["graph_update_time"].append(
            diagnostics.get("graph_update_time", float("nan"))
            if graph_update_time is None
            else graph_update_time
        )
        self.history["peak_server_vram_mb"].append(
            float("nan")
            if peak_server_vram_mb is None
            else peak_server_vram_mb
        )
        self.history["peak_server_reserved_vram_mb"].append(
            float("nan")
            if peak_server_reserved_vram_mb is None
            else peak_server_reserved_vram_mb
        )
        warmup_rounds = int(getattr(self.args, "rq5_warmup_rounds", 0))
        timed_rounds = int(getattr(self.args, "rq5_timed_rounds", 0))
        self.history["is_warmup_round"].append(round_index <= warmup_rounds)
        self.history["is_timed_round"].append(
            warmup_rounds < round_index <= warmup_rounds + timed_rounds
        )
        cumulative = communication_mb + (
            self.history["cumulative_communication_mb"][-1]
            if self.history["cumulative_communication_mb"]
            else 0.0
        )
        self.history["cumulative_communication_mb"].append(cumulative)
        for key in (
            "graph_loss",
            "topk_jaccard",
            "frobenius_distance",
            "spectral_distance",
            "d_task",
            "d_round",
        ):
            # Graph-structure diagnostics are formally sampled only on
            # validation rounds.  Keep non-validation rows explicit as NaN so
            # round-wise history remains aligned without implying a reported
            # measurement was made there.
            if key in {"d_task", "d_round"} and valid_metrics is None:
                value = float("nan")
            else:
                value = diagnostics.get(key, float("nan"))
            self.history[key].append(value)
        self._record_metrics("train", train_metrics)
        self._record_metrics("valid", valid_metrics or {})

    def _snapshot(self):
        return {
            "global_state": clone_state(self.global_state),
            "client_states": {
                key: clone_state(value) for key, value in self.client_states.items()
            },
            "aggregator": (
                self.aggregator.snapshot() if self.args.agg == "fedgl" else None
            ),
        }

    def _restore(self, snapshot):
        self.global_state = clone_state(snapshot["global_state"])
        self.client_states = {
            key: clone_state(value)
            for key, value in snapshot["client_states"].items()
        }
        if snapshot["aggregator"] is not None:
            self.aggregator.restore(snapshot["aggregator"])

    def _save_checkpoint(self, snapshot, best_round, best_rmse):
        if not self.args.save_model:
            return
        output = Path(self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **snapshot,
                "round": best_round,
                "validation_rmse": best_rmse,
                "config": vars(self.args),
            },
            output / "best_model.pt",
        )

    def train(self):
        start = time.perf_counter()
        best_rmse, best_mae, best_round = float("inf"), float("inf"), 0
        best_snapshot, stale_validations = None, 0

        for round_index in range(1, self.args.com_round + 1):
            round_result = self.run_round(round_index)
            if len(round_result) == 5:
                # Keep compatibility with focused tests and legacy callers
                # that monkey-patch run_round with its pre-RQ5 return shape.
                (
                    train,
                    diagnostics,
                    local_time,
                    aggregate_time,
                    communication,
                ) = round_result
                round_result = (
                    train,
                    diagnostics,
                    local_time,
                    aggregate_time,
                    communication,
                    None,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    diagnostics.get("graph_update_time", float("nan")),
                    float("nan"),
                    float("nan"),
                )
            (
                train,
                diagnostics,
                local_time,
                aggregate_time,
                communication,
                participating_clients,
                local_time_mean,
                downlink_mb,
                uplink_mb,
                round_time,
                graph_update_time,
                peak_server_vram_mb,
                peak_server_reserved_vram_mb,
            ) = round_result
            validate_now = (
                round_index == 1
                or round_index % self.args.valid_freq == 0
                or round_index == self.args.com_round
            )
            valid = self.evaluate("valid") if validate_now else None
            self._record_round(
                round_index,
                train,
                valid,
                diagnostics,
                local_time,
                aggregate_time,
                communication,
                participating_clients,
                local_time_mean,
                downlink_mb,
                uplink_mb,
                round_time,
                graph_update_time,
                peak_server_vram_mb,
                peak_server_reserved_vram_mb,
            )

            message = (
                f"Round {round_index:03d}: train RMSE={train.get('RMSE', 0):.6f}"
            )
            if valid:
                message += f", valid RMSE={valid['RMSE']:.6f}"
                if valid["RMSE"] < best_rmse:
                    best_rmse, best_mae, best_round = (
                        valid["RMSE"],
                        valid["MAE"],
                        round_index,
                    )
                    best_snapshot = self._snapshot()
                    self._save_checkpoint(best_snapshot, best_round, best_rmse)
                    stale_validations = 0
                else:
                    stale_validations += 1
            print(message)

            if (
                self.args.enable_early_stop
                and stale_validations >= self.args.early_stop_patience
            ):
                print(f"Early stop at round {round_index}")
                break

        if best_snapshot is None:
            best_snapshot = self._snapshot()
            valid = self.evaluate("valid")
            best_rmse, best_mae = valid["RMSE"], valid["MAE"]
            best_round = self.history["round"][-1]
        self._restore(best_snapshot)
        test_metrics = self.evaluate("test")
        for key, history_key in (("D_task", "d_task"), ("D_round", "d_round")):
            values = np.asarray(self.history[history_key], dtype=float)
            finite = values[np.isfinite(values)]
            test_metrics[key] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
        timed = np.asarray(self.history["is_timed_round"], dtype=bool)

        def _finite_window(key):
            values = np.asarray(self.history[key], dtype=float)
            values = values[timed]
            return values[np.isfinite(values)]

        all_communication = np.asarray(
            self.history["communication_mb"], dtype=float
        )
        all_communication = all_communication[np.isfinite(all_communication)]
        timed_local = _finite_window("local_time_mean")
        timed_server = _finite_window("aggregation_time")
        timed_round = _finite_window("round_time")
        all_round = np.asarray(self.history["round_time"], dtype=float)
        all_round = all_round[np.isfinite(all_round)]
        timed_graph = _finite_window("graph_update_time")
        timed_peak = _finite_window("peak_server_vram_mb")
        timed_reserved = _finite_window("peak_server_reserved_vram_mb")
        best_indices = [
            index
            for index, value in enumerate(self.history["round"])
            if value == best_round
        ]
        if best_indices:
            cumulative_to_best = float(
                self.history["cumulative_communication_mb"][best_indices[-1]]
            )
        else:
            cumulative_to_best = float("nan")
        test_metrics.update(
            {
                "communication_to_best_mb": cumulative_to_best,
                "communication_to_best_gb": cumulative_to_best / 1024,
                "mean_communication_mb_per_round": (
                    float(np.mean(all_communication))
                    if all_communication.size
                    else float("nan")
                ),
                "client_time_mean_s": (
                    float(np.mean(timed_local))
                    if timed_local.size
                    else float("nan")
                ),
                "server_time_mean_s": (
                    float(np.mean(timed_server))
                    if timed_server.size
                    else float("nan")
                ),
                "round_time_mean_s": (
                    float(np.mean(timed_round))
                    if timed_round.size
                    else float("nan")
                ),
                "graph_update_time_mean_s": (
                    float(np.mean(timed_graph))
                    if timed_graph.size
                    else float("nan")
                ),
                "peak_server_vram_mb": (
                    float(np.max(timed_peak))
                    if timed_peak.size
                    else float("nan")
                ),
                "peak_server_reserved_vram_mb": (
                    float(np.max(timed_reserved))
                    if timed_reserved.size
                    else float("nan")
                ),
                "profiled_round_count": int(timed.sum()),
                "total_round_time_s": (
                    float(np.sum(all_round)) if all_round.size else float("nan")
                ),
                "total_wall_time_s": time.perf_counter() - start,
            }
        )
        print(
            f"Best round={best_round}, valid RMSE={best_rmse:.6f}, "
            f"test MAE={test_metrics['MAE']:.6f}, test RMSE={test_metrics['RMSE']:.6f}"
        )
        return {
            "training_history": self.history,
            "best_round": best_round,
            "best_valid_rmse": best_rmse,
            "best_valid_mae": best_mae,
            "final_test_metrics": test_metrics,
            "total_rounds": len(self.history["round"]),
            "total_training_time": time.perf_counter() - start,
        }
