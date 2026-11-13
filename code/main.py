"""FedGL paper-demo entry point."""

import json
import time
from pathlib import Path

import pandas as pd
import torch

from config import parse_args
from core_utils.common_utils import ensure_dir, set_random_seed
from data_processing.fed_dataset_loader import (
    get_federated_dataloaders_for_reliability,
)
from federated.client import FederatedClient
from federated.server import FederatedServer
from graph_construction.graph_builder import (
    build_client_prior,
    client_private_graphs,
    create_context_graphs,
)


def _resolve_paths(args):
    code_dir = Path(__file__).resolve().parent
    dataset = Path(args.dataset_path)
    output = Path(args.output_dir)
    args.dataset_path = str(dataset if dataset.is_absolute() else code_dir / dataset)
    args.output_dir = str(output if output.is_absolute() else code_dir / output)


def _create_clients(
    args,
    train_loaders,
    valid_loaders,
    test_loaders,
    user_graph,
    service_graph,
    metadata,
):
    clients = []
    for client_id in range(args.clients):
        private_user, private_service, local_interactions = client_private_graphs(
            user_graph,
            service_graph,
            train_loaders[client_id],
            metadata["num_users"],
            metadata["num_services"],
        )
        clients.append(
            FederatedClient(
                client_id,
                args,
                train_loaders[client_id],
                valid_loaders[client_id],
                test_loaders[client_id],
                private_user,
                private_service,
                metadata,
                local_interactions=local_interactions,
            )
        )
    return clients


def _save_results(args, results):
    output = ensure_dir(args.output_dir)
    name = f"{args.agg}_seed{args.random_seed}"
    pd.DataFrame(results["training_history"]).to_csv(
        output / f"history_{name}.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "best_round": results["best_round"],
                "best_valid_rmse": results["best_valid_rmse"],
                "best_valid_mae": results["best_valid_mae"],
                **{
                    key.lower(): value
                    for key, value in results["final_test_metrics"].items()
                },
            }
        ]
    ).to_csv(output / f"test_{name}.csv", index=False)
    with (output / f"config_{name}.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)


def run(args):
    _resolve_paths(args)
    set_random_seed(args.random_seed)
    device = torch.device(args.devices)
    print(f"Device={device}, aggregation={args.agg}, seed={args.random_seed}")

    train, valid, test, metadata = get_federated_dataloaders_for_reliability(args)
    (
        user_graph,
        service_graph,
        client_frame,
        peer_frame,
        context_users,
        context_services,
    ) = create_context_graphs(args)
    if (context_users, context_services) != (
        metadata["num_users"],
        metadata["num_services"],
    ):
        raise ValueError("context tables and QoS matrices use different entity counts")

    clients = _create_clients(
        args,
        train,
        valid,
        test,
        user_graph,
        service_graph,
        metadata,
    )
    parameter_count = sum(
        parameter.numel() for parameter in clients[0].model.parameters()
    )
    print(f"Clients={len(clients)}, model parameters={parameter_count:,}")

    prior = None
    if args.agg in {"graph", "graph_v3"} or (
        args.agg == "fedgl" and args.graph_prior_weight > 0
    ):
        prior = build_client_prior(
            args, train, client_frame, peer_frame, device
        )
        print(f"Context prior={tuple(prior.shape)}")

    start = time.time()
    results = FederatedServer(args, clients, prior).train()
    _save_results(args, results)
    print(f"Finished in {(time.time() - start) / 60:.2f} minutes")
    return results


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
