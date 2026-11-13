"""Leakage-free loading and IID/Dirichlet partitioning of the four QoS tasks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


METRIC_FILES = {
    "SR": "SuccessRate",
    "RB": "rightBlock",
    "RH": "recentHeight",
    "RTT": "roundtripTime",
}


class ReliabilityDataset(Dataset):
    def __init__(self, users, services, targets):
        self.users = np.asarray(users, dtype=np.int64)
        self.services = np.asarray(services, dtype=np.int64)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        target = self.targets[index]
        return (
            torch.tensor(self.users[index], dtype=torch.long),
            torch.tensor(self.services[index], dtype=torch.long),
            target[0],
            target[1],
            target[2],
            target[3],
        )


def _read_matrix(path):
    matrix = pd.read_csv(path, sep="\t", header=None).dropna(axis=1, how="all")
    return matrix.to_numpy(dtype=np.float32).clip(min=0)


def _load_metrics(dataset_path, suffix):
    matrices = []
    for task, filename in METRIC_FILES.items():
        path = Path(dataset_path) / f"{filename}{suffix}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = _read_matrix(path)
        if task == "RTT":
            matrix = np.log1p(matrix)
        matrices.append(matrix)
    if len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("the four QoS matrices must have the same shape")
    return np.stack(matrices, axis=-1)


def _make_split_datasets(values, args):
    num_users, num_services, _ = values.shape
    users, services = np.indices((num_users, num_services))
    users, services = users.ravel(), services.ravel()
    targets = values.reshape(-1, 4)

    rng = np.random.default_rng(args.random_seed)
    order = rng.permutation(len(users))
    train_end = int(len(order) * args.train_density)
    valid_end = train_end + int(len(order) * args.valid_density)
    train_index, valid_index, test_index = (
        order[:train_end],
        order[train_end:valid_end],
        order[valid_end:],
    )

    # Fit every normalization scale on training labels only.
    scales = targets[train_index].max(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    targets = targets / scales

    def dataset(index):
        return ReliabilityDataset(users[index], services[index], targets[index])

    metadata = {
        "num_users": num_users,
        "num_services": num_services,
        "scale_factors": dict(zip(METRIC_FILES, scales.tolist())),
    }
    return dataset(train_index), dataset(valid_index), dataset(test_index), metadata


def _largest_remainder(probabilities, size, rng):
    expected = probabilities * size
    counts = np.floor(expected).astype(int)
    remainder = size - counts.sum()
    if remainder:
        score = expected - counts + rng.random(len(counts)) * 1e-12
        counts[np.argsort(score)[-remainder:]] += 1
    return counts


def _partition(dataset, args, proportions, seed):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(args.clients)]

    if args.data_partition == "iid":
        indices = rng.permutation(len(dataset))
        return [
            Subset(dataset, part.tolist())
            for part in np.array_split(indices, args.clients)
        ]

    for user in np.unique(dataset.users):
        indices = np.flatnonzero(dataset.users == user)
        rng.shuffle(indices)
        counts = _largest_remainder(proportions[int(user)], len(indices), rng)
        offset = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(
                indices[offset : offset + count].tolist()
            )
            offset += count
    return [Subset(dataset, indices) for indices in client_indices]


def _make_loaders(datasets, args, shuffle):
    return [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle and len(dataset) > 0,
            num_workers=0,
            pin_memory=args.devices == "cuda",
        )
        for dataset in datasets
    ]


def get_federated_dataloaders_for_reliability(args, logger=print):
    values = _load_metrics(args.dataset_path, args.metric_file_suffix)
    train, valid, test, metadata = _make_split_datasets(values, args)

    rng = np.random.default_rng(args.random_seed)
    proportions = rng.dirichlet(
        np.full(args.clients, args.non_iid_alpha),
        size=metadata["num_users"],
    )
    train_parts = _partition(train, args, proportions, args.random_seed + 1)
    valid_parts = _partition(valid, args, proportions, args.random_seed + 2)
    test_parts = _partition(test, args, proportions, args.random_seed + 3)

    logger(
        f"Data: {values.shape[:2]}, split={len(train)}/{len(valid)}/{len(test)}, "
        f"partition={args.data_partition}, clients={args.clients}"
    )
    return (
        _make_loaders(train_parts, args, True),
        _make_loaders(valid_parts, args, False),
        _make_loaders(test_parts, args, False),
        metadata,
    )
