"""Local proximal objective for FedProx."""


def fedprox_regularization(model, reference, loss, mu):
    """Return loss + mu/2 * ||w - w_global||^2."""
    if reference is None:
        raise ValueError("FedProx requires the current global reference state")
    proximal = sum(
        (parameter - reference[name]).square().sum()
        for name, parameter in model.named_parameters()
    )
    return loss + 0.5 * float(mu) * proximal
