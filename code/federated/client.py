"""One local learner in the federated simulation."""

import time

import torch
from torch import nn
from torch.nn import functional as F

from baseline import (
    apply_hcfgnn_privacy,
    build_specialized_model,
    fedprox_regularization,
)
from core_utils.common_utils import calculate_metrics, clone_state
from models.graph_model import FedGLModel


class FederatedClient:
    def __init__(
        self,
        client_id,
        args,
        train_loader,
        valid_loader,
        test_loader,
        user_graph,
        service_graph,
        metadata,
        local_interactions=None,
    ):
        self.client_id = client_id
        self.args = args
        self.loaders = {
            "train": train_loader,
            "valid": valid_loader,
            "test": test_loader,
        }
        self.device = torch.device(args.devices)
        user_graph = user_graph.to(self.device)
        service_graph = service_graph.to(self.device)
        self.model = build_specialized_model(
            args,
            user_graph,
            service_graph,
            metadata["num_users"],
            metadata["num_services"],
        )
        if self.model is None:
            self.model = FedGLModel(
                args,
                user_graph,
                service_graph,
                metadata["num_users"],
                metadata["num_services"],
                local_interactions=local_interactions,
            )
        self.model = self.model.to(self.device)
        optimizer = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": lambda params, **kw: torch.optim.SGD(params, momentum=0.9, **kw),
        }[args.optimizer]
        self.optimizer = optimizer(
            self.model.parameters(), lr=args.lr, weight_decay=args.decay
        )
        if args.lr_scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, max(args.com_round * args.client_epochs, 1)
            )
        elif args.lr_scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, args.lr_step, gamma=0.5
            )
        else:
            self.scheduler = None
        self.reference = None

    def load_state(self, state, reference=None):
        self.model.load_state_dict(state)
        self.reference = (
            {
                name: value.detach().to(self.device)
                for name, value in reference.items()
            }
            if reference is not None
            else None
        )

    def _prediction_loss(self, predictions, targets):
        if self.args.loss_func == "L1Loss":
            errors = (predictions - targets).abs()
        elif self.args.loss_func == "SmoothL1Loss":
            errors = F.smooth_l1_loss(predictions, targets, reduction="none")
        else:
            errors = (predictions - targets).square()
        if predictions.ndim == 1:
            return errors.mean()
        weights = torch.tensor(
            self.args.task_weights, device=self.device, dtype=errors.dtype
        )
        weights = weights / weights.sum()
        return (errors.mean(dim=0) * weights).sum()

    def _regularization(self, loss):
        if self.reference is None:
            return loss
        if self.args.agg == "fedprox":
            return fedprox_regularization(
                self.model,
                self.reference,
                loss,
                self.args.fedprox_mu,
            )
        if self.args.agg == "fedgl":
            proximal = sum(
                (parameter - self.reference[name]).square().sum()
                for name, parameter in self.model.named_parameters()
            )
            return loss + 0.5 * self.args.fedprox_mu * proximal
        if self.args.agg == "pfedgraph":
            dot = loss.new_zeros(())
            left = loss.new_zeros(())
            right = loss.new_zeros(())
            for name, parameter in self.model.named_parameters():
                reference = self.reference[name]
                dot += (parameter * reference).sum()
                left += parameter.square().sum()
                right += reference.square().sum()
            cosine = dot / (left.sqrt() * right.sqrt()).clamp_min(1e-12)
            return loss - 0.5 * self.args.pfedgraph_lambda * cosine
        return loss

    def _batch(self, batch):
        user, service, *targets = (item.to(self.device) for item in batch)
        target = targets[0] if self.args.task_mode == "single" else torch.stack(targets, 1)
        return user, service, target

    def train(self):
        loader = self.loaders["train"]
        if len(loader.dataset) == 0:
            return clone_state(self.model.state_dict()), {"n": 0, "loss": 0.0}, 0.0

        start = time.time()
        self.model.train()
        self.model.clear_cache()
        last_predictions, last_targets, last_loss = [], [], 0.0
        for _ in range(self.args.client_epochs):
            last_predictions, last_targets, last_loss = [], [], 0.0
            for batch in loader:
                user, service, target = self._batch(batch)
                self.optimizer.zero_grad()
                prediction = self.model(user, service)
                loss = self._regularization(self._prediction_loss(prediction, target))
                loss.backward()
                if self.args.agg == "hcfgnn":
                    apply_hcfgnn_privacy(
                        self.model.parameters(),
                        clip=self.args.max_gradient,
                        noise_multiplier=self.args.hcfgnn_ldp_lambda,
                    )
                elif self.args.max_gradient > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_gradient
                    )
                self.optimizer.step()
                last_loss += float(loss.detach()) * len(user)
                last_predictions.append(prediction.detach().cpu())
                last_targets.append(target.detach().cpu())
            if self.scheduler:
                self.scheduler.step()

        metrics = calculate_metrics(
            torch.cat(last_predictions), torch.cat(last_targets)
        )
        metrics["loss"] = last_loss / metrics["n"]
        return clone_state(self.model.state_dict()), metrics, time.time() - start

    def evaluate(self, split):
        loader = self.loaders[split]
        if len(loader.dataset) == 0:
            return {"n": 0, "loss": 0.0}
        self.model.eval()
        self.model.prepare_eval()
        predictions, targets, loss_sum = [], [], 0.0
        with torch.no_grad():
            for batch in loader:
                user, service, target = self._batch(batch)
                prediction = self.model(user, service, use_cache=True)
                loss_sum += float(self._prediction_loss(prediction, target)) * len(user)
                predictions.append(prediction.cpu())
                targets.append(target.cpu())
        metrics = calculate_metrics(torch.cat(predictions), torch.cat(targets))
        metrics["loss"] = loss_sum / metrics["n"]
        return metrics

    @property
    def train_size(self):
        return len(self.loaders["train"].dataset)
