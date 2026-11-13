"""Context graphs, client-private subgraphs, and the client prior graph."""

from collections import Counter, defaultdict
from pathlib import Path

import dgl
import pandas as pd
import torch
import torch.nn.functional as F


def _context_table(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _resolve_columns(frame, requested):
    available = {column.casefold(): column for column in frame.columns}
    columns = [available[name.casefold()] for name in requested if name.casefold() in available]
    if not columns:
        raise ValueError(f"no requested context columns found in {list(frame.columns)}")
    return columns


def context_graph(frame, requested_columns):
    """Create an entity-context graph with entity IDs placed first."""
    columns = _resolve_columns(frame, requested_columns)
    num_entities = len(frame)
    feature_ids = {}
    src, dst = [], []

    for entity_id, row in frame.iterrows():
        for column in columns:
            value = str(row[column]) if pd.notna(row[column]) else "__MISSING__"
            key = (column, value)
            if key not in feature_ids:
                feature_ids[key] = num_entities + len(feature_ids)
            context_id = feature_ids[key]
            src.extend([entity_id, context_id])
            dst.extend([context_id, entity_id])

    graph = dgl.graph((src, dst), num_nodes=num_entities + len(feature_ids))
    return dgl.add_self_loop(graph), num_entities


def create_context_graphs(args):
    client_path = Path(args.dataset_path) / args.client_ctx_file
    peer_path = Path(args.dataset_path) / args.peer_ctx_file
    client_frame, peer_frame = _context_table(client_path), _context_table(peer_path)
    user_graph, num_users = context_graph(client_frame, args.client_feature_cols)
    service_graph, num_services = context_graph(peer_frame, args.peer_feature_cols)
    return user_graph, service_graph, client_frame, peer_frame, num_users, num_services


def dataset_pairs(dataset):
    for index in range(len(dataset)):
        sample = dataset[index]
        yield int(sample[0]), int(sample[1])


def private_context_graph(graph, local_entities, num_entities):
    """Retain only context edges incident to locally observed entities."""
    src, dst = graph.edges()
    local = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=graph.device)
    if local_entities:
        local[list(sorted(local_entities))] = True
    keep = (src == dst) | local[src] | local[dst]
    return dgl.graph(
        (src[keep], dst[keep]),
        num_nodes=graph.num_nodes(),
        idtype=graph.idtype,
        device=graph.device,
    )


def client_private_graphs(user_graph, service_graph, train_loader, num_users, num_services):
    """Return private context graphs and training-only user-service calls."""
    pairs = set(dataset_pairs(train_loader.dataset))
    users = {user for user, _ in pairs}
    services = {service for _, service in pairs}
    return (
        private_context_graph(user_graph, users, num_users),
        private_context_graph(service_graph, services, num_services),
        tuple(sorted(pairs)),
    )


def projected_entity_graph(graph, num_entities):
    """Entity-context-entity semantic view used by the compact HAN."""
    src, dst = graph.edges()
    groups = defaultdict(set)
    for left, right in zip(src.cpu().tolist(), dst.cpu().tolist()):
        if left < num_entities <= right:
            groups[right].add(left)

    edges = {(node, node) for node in range(graph.num_nodes())}
    for entities in groups.values():
        edges.update((left, right) for left in entities for right in entities)
    source, target = zip(*edges)
    return dgl.graph(
        (list(source), list(target)),
        num_nodes=graph.num_nodes(),
        device=graph.device,
    )


def _node_counts(train_loaders):
    users, services = [], []
    for loader in train_loaders:
        user_count, service_count = Counter(), Counter()
        for user, service in dataset_pairs(loader.dataset):
            user_count[user] += 1
            service_count[service] += 1
        users.append(user_count)
        services.append(service_count)
    return users, services


def _context_similarity(counts, frame, requested_columns):
    similarities = []
    for column in _resolve_columns(frame, requested_columns):
        values = frame[column].astype("string").fillna("__MISSING__")
        categories = {value: index for index, value in enumerate(sorted(values.unique()))}
        profile = torch.zeros(len(counts), len(categories))
        for client_id, counter in enumerate(counts):
            for node_id, count in counter.items():
                profile[client_id, categories[values.iloc[node_id]]] += count
        profile = F.normalize(profile, dim=1)
        similarities.append(profile @ profile.T)
    return torch.stack(similarities).mean(0)


def build_client_prior(args, train_loaders, client_frame, peer_frame, device):
    """Build [4, M, M] from training partition IDs and non-label context only."""
    user_counts, service_counts = _node_counts(train_loaders)
    prior = 0.5 * (
        _context_similarity(user_counts, client_frame, args.client_feature_cols)
        + _context_similarity(service_counts, peer_frame, args.peer_feature_cols)
    )
    prior = 0.5 * (prior + prior.T)
    prior.fill_diagonal_(1.0)
    prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
    return prior.to(device).unsqueeze(0).repeat(4, 1, 1)
