"""Base-layer aggregation with client-private prediction heads."""

from baseline.FedAvg import weighted_average
from core_utils.common_utils import clone_state


def _is_personalization_parameter(key):
    return "head" in key.casefold()


class FedPerAggregator:
    """Aggregate the shared backbone and retain each client's output heads."""

    def __init__(self, args):
        del args

    @staticmethod
    def prepare_client_state(personalized_state, global_state):
        """Combine the newest shared base with the client's private heads."""
        result = clone_state(global_state)
        for key, value in personalized_state.items():
            if _is_personalization_parameter(key):
                result[key] = value.clone()
        return result

    def aggregate(self, states, client_ids, sample_sizes, reference):
        del reference
        if not states:
            raise ValueError("FedPer requires at least one client state")
        average = weighted_average(states, sample_sizes)
        personalized = {}
        for client_id, state in zip(client_ids, states):
            result = clone_state(average)
            for key, value in state.items():
                if _is_personalization_parameter(key):
                    result[key] = value.clone()
            personalized[client_id] = result
        return average, personalized
