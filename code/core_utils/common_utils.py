"""Small utilities shared by the research demo."""

import copy
import random
from pathlib import Path

import numpy as np
import torch


TASKS = ("SR", "RB", "RH", "RTT")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state(state, device="cpu"):
    return {
        key: value.detach().clone().to(device)
        if torch.is_tensor(value)
        else copy.deepcopy(value)
        for key, value in state.items()
    }


def state_bytes(state):
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if torch.is_tensor(value)
    )


def calculate_metrics(predictions, targets):
    predictions = predictions.detach().float().cpu()
    targets = targets.detach().float().cpu()
    if predictions.ndim == 1:
        predictions = predictions[:, None]
        targets = targets[:, None]

    metrics = {"n": int(targets.shape[0])}
    maes = (predictions - targets).abs().mean(dim=0)
    rmses = ((predictions - targets).square().mean(dim=0)).sqrt()
    names = TASKS[: predictions.shape[1]]
    for index, name in enumerate(names):
        metrics[f"{name}_MAE"] = float(maes[index])
        metrics[f"{name}_RMSE"] = float(rmses[index])
    metrics["MAE"] = float(maes.mean())
    metrics["RMSE"] = float(rmses.mean())
    return metrics


def combine_client_metrics(metrics):
    """Combine client MAE/RMSE exactly from their sample counts."""
    metrics = [item for item in metrics if item.get("n", 0) > 0]
    if not metrics:
        return {"n": 0, "loss": 0.0, "MAE": 0.0, "RMSE": 0.0}

    total = sum(item["n"] for item in metrics)
    result = {"n": total}
    if all("loss" in item for item in metrics):
        result["loss"] = sum(item["loss"] * item["n"] for item in metrics) / total

    task_names = [
        task for task in TASKS if f"{task}_RMSE" in metrics[0]
    ]
    for task in task_names:
        result[f"{task}_MAE"] = sum(
            item[f"{task}_MAE"] * item["n"] for item in metrics
        ) / total
        result[f"{task}_RMSE"] = (
            sum(item[f"{task}_RMSE"] ** 2 * item["n"] for item in metrics)
            / total
        ) ** 0.5
    result["MAE"] = float(np.mean([result[f"{task}_MAE"] for task in task_names]))
    result["RMSE"] = float(np.mean([result[f"{task}_RMSE"] for task in task_names]))
    return result


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
