"""FedProx uses FedAvg aggregation after proximal local optimization."""

from baseline.FedAvg import FedAvgAggregator


class FedProxAggregator(FedAvgAggregator):
    """The server step of FedProx is sample-weighted averaging."""

