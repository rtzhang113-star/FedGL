"""FedProx baseline exports."""

from .aggregator import FedProxAggregator
from .regularizer import fedprox_regularization

__all__ = ["FedProxAggregator", "fedprox_regularization"]
