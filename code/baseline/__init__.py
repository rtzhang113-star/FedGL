"""Adapters for paper baselines implemented inside the FedGL experiment stack."""

from importlib import import_module

from .FedAtt import FedAttAggregator
from .FedAvg import FedAvgAggregator
from .FedPer import FedPerAggregator
from .FedProx import FedProxAggregator, fedprox_regularization
from .pFedGraph import PFedGraphAggregator
from .pFedLN import PFedLNAggregator


def _hcfgnn_module():
    # The directory name mirrors the paper name requested by the experiment
    # layout. import_module can load it even though '-' is not valid in a
    # regular Python import statement.
    return import_module(".HC-FGNN", __name__)


def _flmfgm_module():
    return import_module(".FL-MFGM", __name__)


def build_specialized_aggregator(args, prior=None):
    """Return a paper-specific aggregator, or None for built-in baselines."""
    del prior
    if args.agg in {"avg", "fedavg"}:
        return FedAvgAggregator(args)
    if args.agg in {"att", "fedatt"}:
        return FedAttAggregator(args)
    if args.agg == "fedprox":
        return FedProxAggregator(args)
    if args.agg in {"per", "fedper"}:
        return FedPerAggregator(args)
    if args.agg == "pfedgraph":
        return PFedGraphAggregator(args)
    if args.agg == "pfedln":
        return PFedLNAggregator(args)
    if args.agg == "hcfgnn":
        return _hcfgnn_module().HCFGNNAggregator(args)
    if args.agg == "flmfgm":
        return _flmfgm_module().FLMFGMAggregator(args)
    if args.agg == "fncf":
        return import_module(".FNCF", __name__).FNCFAggregator(args)
    return None


def build_specialized_model(
    args,
    user_graph,
    service_graph,
    num_users,
    num_services,
):
    """Return a method-specific local model, or None for the shared backbone."""
    if args.agg == "hcfgnn":
        return _hcfgnn_module().HCFGNNModel(
            args,
            user_graph,
            service_graph,
            num_users,
            num_services,
        )
    if args.agg == "flmfgm":
        return _flmfgm_module().FLMFGMModel(args, num_users, num_services)
    if args.agg == "fncf":
        return import_module(".FNCF", __name__).FNCFModel(
            args,
            user_graph,
            num_users,
            num_services,
        )
    return None


def apply_hcfgnn_privacy(parameters, clip, noise_multiplier):
    """Apply the optional HC-FGNN local-DP gradient transform."""
    return _hcfgnn_module().apply_local_dp(
        parameters,
        clip=clip,
        noise_multiplier=noise_multiplier,
    )


__all__ = [
    "FedAttAggregator",
    "FedAvgAggregator",
    "FedPerAggregator",
    "FedProxAggregator",
    "PFedGraphAggregator",
    "PFedLNAggregator",
    "apply_hcfgnn_privacy",
    "build_specialized_aggregator",
    "build_specialized_model",
    "fedprox_regularization",
]
