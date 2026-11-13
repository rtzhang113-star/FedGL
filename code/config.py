"""Command-line configuration for the FedGL research demo."""

import argparse

import torch


# These are the seven configurations used by the current RQ2 protocol.  Keep
# this tuple canonical: experiment launchers can use it to construct the
# frozen matrix without accidentally including diagnostics from RQ4/RQ2-6.
RQ2_1_VARIANTS = (
    "full",
    "w_o_context_encoding",
    "symmetric_head",
    "static_graph",
    "uniform_aggregation",
    "w_o_context_encoding_symmetric_head",
    "static_graph_uniform_aggregation",
)

# The old command-line names are accepted for existing one-off runs, but are
# deliberately not part of RQ2_1_VARIANTS and therefore cannot enter the new
# RQ2 matrix through code that iterates the canonical tuple.
RQ2_1_LEGACY_ALIASES = {
    "w_o_heterogeneous_encoder": "w_o_context_encoding",
    "w_o_asymmetric_head": "symmetric_head",
}
RQ2_1_LEGACY_VARIANTS = (
    "w_o_heterogeneous_encoder",
    "w_o_asymmetric_head",
    # These two controls belong to other research questions, but remain
    # accepted as legacy switches so older command files still parse.
    "w_o_prior_graph",
    "shared_graph",
)
RQ2_1_VARIANT_CHOICES = RQ2_1_VARIANTS + RQ2_1_LEGACY_VARIANTS

# RQ3 is split into two paper tables, but ``full`` is the same canonical
# configuration in both.  Keep the per-table tuples for experiment launchers
# and expose a de-duplicated union for the single ``--rq3_variant`` switch.
RQ3_1_VARIANTS = (
    "full",
    "w_o_interaction",
    "gat",
    "w_o_context_encoding",
)
RQ3_2_VARIANTS = (
    "full",
    "symmetric_head",
    "single_task",
)
RQ3_VARIANT_CHOICES = tuple(dict.fromkeys(RQ3_1_VARIANTS + RQ3_2_VARIANTS))

# RQ4 reuses ``full`` across the task-graph and layerwise-personalization
# tables.  Keep the table-specific tuples for launchers and expose one
# de-duplicated command-line choice list.
RQ4_1_VARIANTS = (
    "full",
    "shared_graph",
    "w_o_dynamic_graph",
    "w_o_context_prior_graph",
)
RQ4_2_VARIANTS = (
    "full",
    "uniform_aggregation",
    "w_o_residual_fusion",
)
RQ4_VARIANT_CHOICES = tuple(dict.fromkeys(RQ4_1_VARIANTS + RQ4_2_VARIANTS))

# RQ5 has three measurement modes.  The method sets are kept explicit so a
# launcher cannot silently put a convergence-only baseline into the overhead
# or scalability tables (or vice versa).
RQ5_MODE_CHOICES = ("convergence", "overhead", "scalability")
RQ5_CONVERGENCE_METHODS = (
    "avg",
    "fedavg",
    "fedprox",
    "per",
    "fedper",
    "pfedln",
    "pfedgraph",
    "fedgl",
)
RQ5_SYSTEM_METHODS = ("avg", "fedavg", "pfedgraph", "fedgl")
RQ5_METHODS_BY_MODE = {
    "convergence": RQ5_CONVERGENCE_METHODS,
    "overhead": RQ5_SYSTEM_METHODS,
    "scalability": RQ5_SYSTEM_METHODS,
}

# RQ6 varies one of the three paper-defined FedGL hyperparameters at a time.
# The code-level names are graph_prior_weight (rho), serverbeta (beta), and
# graph_evolution_epochs (E_g); keep the paper values centralized so launchers
# and tests cannot silently introduce an extra sensitivity point.
RQ6_RHO_VALUES = (0.0, 0.05, 0.1, 0.3, 0.5)
RQ6_BETA_VALUES = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
RQ6_EG_VALUES = (1, 5, 10, 20)
RQ6_GRAPH_STEPS_VALUES = RQ6_EG_VALUES
RQ6_AXIS_CHOICES = ("rho", "beta", "graph_steps")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _apply_rq2_1_variant(args):
    """Resolve a named RQ2 preset while retaining lower-level compatibility.

    A named preset is authoritative for the RQ2 factors.  In particular, a
    stale launcher flag cannot silently turn a Full run into a different
    configuration.  ``use_graph_encoder`` remains the old low-level switch;
    ``use_context_encoding`` distinguishes the paper-faithful direct-embedding
    ablation from the legacy feed-forward replacement used by old callers.
    """
    if args.rq2_1_variant is None:
        return

    # Canonical Full-FedGL settings.  Only RQ2 factors are reset here; unrelated
    # hyperparameters supplied by the caller remain untouched.
    args.use_context_encoding = True
    args.use_graph_encoder = True
    args.use_interaction_fusion = True
    args.use_asymmetric_head = True
    args.graph_prior_weight = 0.1
    args.task_graph_mode = "task_specific"
    args.freeze_task_graphs = False
    args.layer_routing_mode = "task_specific"

    variant = RQ2_1_LEGACY_ALIASES.get(args.rq2_1_variant, args.rq2_1_variant)
    if variant == "w_o_context_encoding":
        # The model uses this explicit switch to skip both context graph
        # encoding and local interaction fusion, returning raw embeddings.
        args.use_context_encoding = False
        args.use_graph_encoder = False
        args.use_interaction_fusion = False
    elif variant == "symmetric_head":
        args.use_asymmetric_head = False
    elif variant == "w_o_prior_graph":
        args.graph_prior_weight = 0.0
    elif variant == "shared_graph":
        args.task_graph_mode = "shared"
    elif variant == "static_graph":
        args.freeze_task_graphs = True
    elif variant == "uniform_aggregation":
        args.layer_routing_mode = "uniform_conv"
    elif variant == "w_o_context_encoding_symmetric_head":
        args.use_context_encoding = False
        args.use_graph_encoder = False
        args.use_interaction_fusion = False
        args.use_asymmetric_head = False
    elif variant == "static_graph_uniform_aggregation":
        args.freeze_task_graphs = True
        args.layer_routing_mode = "uniform_conv"


def _apply_rq3_variant(args):
    """Resolve a canonical RQ3 client-side preset.

    RQ3 compares client representation/head structures while keeping the
    server-side mechanism fixed.  The preset therefore resets every factor
    used by the two RQ3 tables, including the lower-level compatibility
    switches, before applying the requested variant.
    """
    if args.rq3_variant is None:
        return

    # Full RQ3 configuration: multi-task, asymmetric head, dual-branch HAN,
    # context encoding, and private interaction fusion.  Server settings are
    # also made explicit so stale launcher flags cannot alter the matrix.
    args.task_mode = "multi"
    args.task_weights = [1.0, 1.0, 1.0, 1.0]
    args.use_asymmetric_head = True
    args.use_han = True
    args.use_context_encoding = True
    args.use_graph_encoder = True
    args.use_interaction_fusion = True
    args.task_graph_mode = "task_specific"
    args.freeze_task_graphs = False
    args.layer_routing_mode = "task_specific"
    args.graph_prior_weight = 0.1

    if args.rq3_variant == "w_o_interaction":
        # Keep the HAN/GAT context encoder and remove only the interaction
        # attention plus semantic fusion path.
        args.use_interaction_fusion = False
    elif args.rq3_variant == "gat":
        args.use_han = False
    elif args.rq3_variant == "w_o_context_encoding":
        # Direct trainable embeddings imply no interaction representation
        # fusion; the model enforces this invariant as well.
        args.use_context_encoding = False
        args.use_graph_encoder = False
        args.use_interaction_fusion = False
    elif args.rq3_variant == "symmetric_head":
        args.use_asymmetric_head = False
    elif args.rq3_variant == "single_task":
        args.task_mode = "single"
        args.task_weights = [1.0, 0.0, 0.0, 0.0]
    elif args.rq3_variant != "full":
        # ``argparse`` normally catches this through ``choices``.  Keep the
        # helper defensive for direct callers and tests using a namespace.
        raise ValueError(f"unknown RQ3 variant: {args.rq3_variant}")

    args.active_task_count = 1 if args.task_mode == "single" else 4


def _apply_rq4_variant(args):
    """Resolve an authoritative RQ4 server-side preset.

    RQ4 varies only server-side graph/personalization factors.  Reset all
    client and server factors that define the paper's Full-FedGL condition so
    stale flags from an earlier command cannot contaminate a named run.
    ``non_iid_alpha`` is intentionally left untouched because it is the RQ4-3
    matrix axis rather than a preset factor.
    """
    if args.rq4_variant is None:
        return

    # Paper-faithful Full configuration.
    args.task_mode = "multi"
    args.task_weights = [1.0, 1.0, 1.0, 1.0]
    args.active_task_count = 4
    args.use_asymmetric_head = True
    args.use_han = True
    args.use_context_encoding = True
    args.use_graph_encoder = True
    args.use_interaction_fusion = True
    args.graph_generation_method = "gnn"
    args.task_graph_mode = "task_specific"
    args.freeze_task_graphs = False
    args.graph_prior_weight = 0.1
    args.layer_routing_mode = "task_specific"
    args.serverbeta = 0.3

    args.static_graph_init_source = "not_applicable"
    if args.rq4_variant == "shared_graph":
        args.task_graph_mode = "shared"
    elif args.rq4_variant == "w_o_dynamic_graph":
        args.freeze_task_graphs = True
        # The current static implementation initializes from the learned
        # context-prior graph when rho is non-zero.
        if args.graph_prior_weight > 0:
            args.static_graph_init_source = "context_prior"
    elif args.rq4_variant == "w_o_context_prior_graph":
        args.graph_prior_weight = 0.0
    elif args.rq4_variant == "uniform_aggregation":
        args.layer_routing_mode = "uniform_conv"
    elif args.rq4_variant == "w_o_residual_fusion":
        args.serverbeta = 1.0
    elif args.rq4_variant != "full":
        # ``argparse`` normally catches this through ``choices``.  Keep the
        # helper defensive for direct callers and namespace-based tests.
        raise ValueError(f"unknown RQ4 variant: {args.rq4_variant}")


def _apply_rq5_mode(args):
    """Apply the paper-authoritative RQ5 protocol for a named mode.

    RQ5 is a measurement protocol rather than a model ablation.  Its common
    data, participation, training, and checkpoint-selection settings are
    therefore reset together.  Scalability is the only mode whose client
    count remains a command-line axis.  FedGL itself is reset to its canonical
    Full configuration so stale RQ2--RQ4 switches cannot contaminate timing
    comparisons.
    """
    if args.rq5_mode is None:
        return

    args.metric_file_suffix = "_12_1000"
    args.train_density = 0.3
    args.valid_density = 0.2
    args.data_partition = "non_iid"
    args.non_iid_alpha = 0.5
    args.client_frac = 0.7
    args.com_round = 500
    args.client_epochs = 5
    args.valid_freq = 5
    args.enable_early_stop = False
    args.task_mode = "multi"
    args.task_weights = [1.0, 1.0, 1.0, 1.0]
    args.active_task_count = 4
    if args.rq5_mode in {"convergence", "overhead"}:
        args.clients = 100

    if args.agg == "fedgl":
        # Canonical Full-FedGL controls used by RQ4 and the main method.
        args.use_asymmetric_head = True
        args.use_han = True
        args.use_context_encoding = True
        args.use_graph_encoder = True
        args.use_interaction_fusion = True
        args.graph_generation_method = "gnn"
        args.task_graph_mode = "task_specific"
        args.freeze_task_graphs = False
        args.graph_prior_weight = 0.1
        args.layer_routing_mode = "task_specific"
        args.serverbeta = 0.3


def _apply_rq6_axis(args):
    """Resolve one paper-authoritative RQ6 sensitivity axis.

    ``--rq6_axis`` selects which existing low-level CLI value is allowed to
    vary.  The selected value is captured before the complete RQ6 protocol is
    restored, so stale flags cannot alter any non-selected factor.  RQ6 uses
    the existing code-level names for the paper symbols rho, beta, and E_g.
    """
    if args.rq6_axis is None:
        return

    axis_fields = {
        "rho": ("graph_prior_weight", RQ6_RHO_VALUES),
        "beta": ("serverbeta", RQ6_BETA_VALUES),
        "graph_steps": ("graph_evolution_epochs", RQ6_EG_VALUES),
    }
    try:
        field, allowed = axis_fields[args.rq6_axis]
    except KeyError as exc:
        raise ValueError(f"unknown RQ6 axis: {args.rq6_axis}") from exc

    selected_value = getattr(args, field)
    if args.rq6_axis == "graph_steps":
        if selected_value <= 0:
            raise ValueError("RQ6 graph_steps must be positive")
        if selected_value not in allowed:
            raise ValueError(
                "RQ6 graph_steps must be one of: "
                + ", ".join(str(value) for value in allowed)
            )
    else:
        if selected_value not in allowed:
            symbol = "rho" if args.rq6_axis == "rho" else "beta"
            raise ValueError(
                f"RQ6 {symbol} must be one of: "
                + ", ".join(str(value) for value in allowed)
            )

    # Paper protocol: Setting 2 (_12_1000), d=0.3, alpha_NIID=0.5, M=100, 70%
    # participation, 500 rounds, five local epochs, and validation-driven
    # early stopping with patience 30.
    args.metric_file_suffix = "_12_1000"
    args.train_density = 0.3
    args.valid_density = 0.2
    args.data_partition = "non_iid"
    args.non_iid_alpha = 0.5
    args.clients = 100
    args.client_frac = 0.7
    args.com_round = 500
    args.client_epochs = 5
    args.valid_freq = 5
    args.enable_early_stop = True
    args.early_stop_patience = 30
    args.rq5_warmup_rounds = 10
    args.rq5_timed_rounds = 30

    # Full FedGL client and server configuration.  These assignments are
    # intentionally authoritative, including fields introduced by RQ4, so a
    # copied command line cannot carry an old ablation into RQ6.
    args.task_mode = "multi"
    args.task_weights = [1.0, 1.0, 1.0, 1.0]
    args.active_task_count = 4
    args.use_asymmetric_head = True
    args.use_han = True
    args.use_context_encoding = True
    args.use_graph_encoder = True
    args.use_interaction_fusion = True
    args.graph_generation_method = "gnn"
    args.task_graph_mode = "task_specific"
    args.freeze_task_graphs = False
    args.graph_prior_weight = 0.1
    args.layer_routing_mode = "task_specific"
    args.serverbeta = 0.3
    args.graph_evolution_epochs = 10
    args.static_graph_init_source = "not_applicable"

    # Restore only the selected sensitivity value.  The RQ5 timing-window
    # controls are reset explicitly to their documented defaults because the
    # RQ6 E_g runner may reuse them.
    setattr(args, field, selected_value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FedGL: task-specific graph federated learning"
    )

    # Common experiment setting
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--client_ctx_file", default="ClientWithCTX.csv")
    parser.add_argument("--peer_ctx_file", default="PeerWithCTX.csv")
    parser.add_argument(
        "--client_feature_cols",
        nargs="+",
        default=["as", "country", "isp", "timezone"],
    )
    parser.add_argument(
        "--peer_feature_cols",
        nargs="+",
        default=["as", "country", "isp", "timezone"],
    )
    parser.add_argument("--metric_file_suffix", default="_12_1000")
    parser.add_argument("--train_density", type=float, default=0.1)
    parser.add_argument("--valid_density", type=float, default=0.2)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--client_frac", type=float, default=0.7)
    parser.add_argument(
        "--data_partition", choices=["iid", "non_iid"], default="non_iid"
    )
    parser.add_argument("--non_iid_alpha", type=float, default=0.5)
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--random_seed", type=int, default=3407)

    # RQ1: baselines
    parser.add_argument(
        "--agg",
        choices=[
            "local",
            "avg",
            "fedavg",
            "att",
            "fedatt",
            "fedprox",
            "per",
            "fedper",
            "krum",
            "pfedgraph",
            "pfedln",
            "hcfgnn",
            "flmfgm",
            "fncf",
            "graph",
            "graph_v2",
            "graph_v3",
            "fedgl",
        ],
        default="fedgl",
    )
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--fedatt_stepsize", type=float, default=1.2)
    parser.add_argument("--krum_f", type=int, default=1)
    parser.add_argument("--pfedgraph_alpha", type=float, default=0.8)
    parser.add_argument("--pfedgraph_lambda", type=float, default=0.1)
    parser.add_argument(
        "--pfedgraph_similarity_scope",
        choices=["all", "head"],
        default="head",
    )
    parser.add_argument(
        "--pfedgraph_vector_size",
        type=int,
        default=8192,
        help="maximum pooled parameter-vector length used for similarity",
    )
    parser.add_argument("--pfedln_neighbors", type=int, default=10)
    parser.add_argument("--pfedln_temperature", type=float, default=0.5)
    parser.add_argument(
        "--pfedln_mix",
        type=float,
        default=1.0,
        help="neighbor aggregate weight for pFedLN base layers",
    )
    parser.add_argument("--pfedln_vector_size", type=int, default=8192)
    parser.add_argument("--hcfgnn_neighbors", type=int, default=10)
    parser.add_argument("--hcfgnn_attention_heads", type=int, default=3)
    parser.add_argument(
        "--hcfgnn_use_graph_encoder",
        type=str2bool,
        default=False,
        help="encode HC-FGNN user/service and neighbour embeddings with each client's graphs",
    )
    parser.add_argument(
        "--hcfgnn_ldp_lambda",
        type=float,
        default=0.0,
        help="adaptive Laplace gradient-noise multiplier; 0 disables LDP noise",
    )
    parser.add_argument(
        "--flmfgm_upload_ratio",
        type=float,
        default=0.01,
        help="fraction of largest peer-factor delta entries uploaded per client",
    )
    parser.add_argument("--flmfgm_momentum", type=float, default=0.9)
    parser.add_argument(
        "--fncf_hidden_dims",
        nargs="+",
        type=int,
        default=[128, 64, 32],
    )

    # =====================================================================
    # RQ2: main-component and two-factor ablations
    # Recommended experiment interface:
    #   --rq2_1_variant full
    #   --rq2_1_variant w_o_context_encoding
    #   --rq2_1_variant symmetric_head
    #   --rq2_1_variant static_graph
    #   --rq2_1_variant uniform_aggregation
    #   --rq2_1_variant w_o_context_encoding_symmetric_head
    #   --rq2_1_variant static_graph_uniform_aggregation
    # When omitted, the lower-level switches below remain backward compatible.
    parser.add_argument(
        "--rq2_1_variant", choices=RQ2_1_VARIANT_CHOICES, default=None
    )
    parser.add_argument(
        "--rq3_variant",
        choices=RQ3_VARIANT_CHOICES,
        default=None,
        help=(
            "canonical RQ3 client-side preset; use one name for either "
            "the local-representation or prediction-head table"
        ),
    )
    parser.add_argument(
        "--rq4_variant",
        choices=RQ4_VARIANT_CHOICES,
        default=None,
        help="canonical RQ4 server-side graph/personalization preset",
    )
    parser.add_argument(
        "--rq5_mode",
        choices=RQ5_MODE_CHOICES,
        default=None,
        help="paper-authoritative RQ5 convergence, overhead, or scalability protocol",
    )
    parser.add_argument(
        "--rq6_axis",
        choices=RQ6_AXIS_CHOICES,
        default=None,
        help="paper-authoritative RQ6 sensitivity axis: rho, beta, or graph_steps",
    )
    parser.add_argument(
        "--use_context_encoding",
        type=str2bool,
        default=None,
        help=(
            "paper-faithful context path switch; false uses direct trainable "
            "user/service embeddings (named RQ2 presets set this explicitly)"
        ),
    )
    parser.add_argument(
        "--use_graph_encoder",
        type=str2bool,
        default=True,
        help="false uses a parameter-matched feed-forward encoder without messages",
    )
    parser.add_argument(
        "--use_interaction_fusion",
        type=str2bool,
        default=True,
        help=(
            "enable private bipartite interaction attention and semantic fusion "
            "after context encoding"
        ),
    )
    parser.add_argument(
        "--use_asymmetric_head",
        type=str2bool,
        default=True,
        help="false gives the SR and auxiliary branches identical [128, 64] widths",
    )
    parser.add_argument(
        "--graph_prior_weight",
        type=float,
        default=0.1,
        help="0 removes the prior from graph learning and embedding aggregation",
    )
    parser.add_argument(
        "--task_graph_mode",
        choices=["task_specific", "shared"],
        default="task_specific",
    )
    parser.add_argument(
        "--freeze_task_graphs",
        type=str2bool,
        default=False,
        help="freeze four full-client task graphs at initialization",
    )
    parser.add_argument(
        "--layer_routing_mode",
        choices=[
            "task_specific",
            "uniform_conv",
            "uniform_att",
            "local",
            "heuristic",
        ],
        default="task_specific",
        help="uniform_conv is the RQ2-1 uniform-aggregation ablation",
    )

    # RQ2-6: task-graph structural diagnostics
    parser.add_argument("--task_graph_topk", type=int, default=5)

    # RQ4: client encoder and server graph generator
    parser.add_argument("--use_han", type=str2bool, default=True)
    parser.add_argument(
        "--graph_generation_method", choices=["mlp", "gnn"], default="gnn"
    )

    # RQ5 and RQ6
    parser.add_argument("--com_round", type=int, default=500)
    parser.add_argument("--valid_freq", type=int, default=5)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument(
        "--rq5_warmup_rounds",
        type=int,
        default=10,
        help="RQ5 profiling warm-up rounds (can be overridden for preflight)",
    )
    parser.add_argument(
        "--rq5_timed_rounds",
        type=int,
        default=30,
        help="RQ5 profiling rounds after warm-up (can be overridden for preflight)",
    )

    # Local model and optimization
    parser.add_argument("--task_mode", choices=["single", "multi"], default="multi")
    parser.add_argument(
        "--task_weights", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0]
    )
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=0.00035)
    parser.add_argument(
        "--loss_func",
        choices=["MSELoss", "L1Loss", "SmoothL1Loss"],
        default="MSELoss",
    )
    parser.add_argument(
        "--optimizer", choices=["AdamW", "Adam", "SGD"], default="AdamW"
    )
    parser.add_argument("--max_gradient", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type", choices=["none", "cosine", "step"], default="none"
    )
    parser.add_argument("--lr_step", type=int, default=50)

    # FedGL graph learning and personalized fusion
    parser.add_argument("--serverbeta", type=float, default=0.3)
    parser.add_argument("--client_feature_dim", type=int, default=128)
    parser.add_argument("--meta_graph_hidden_dim", type=int, default=64)
    parser.add_argument("--task_graph_lr", type=float, default=0.005)
    parser.add_argument("--graph_evolution_epochs", type=int, default=10)
    parser.add_argument("--graph_sparsity", type=float, default=0.05)
    parser.add_argument("--graph_smoothness", type=float, default=0.005)
    parser.add_argument("--attention_temperature", type=float, default=0.1)

    # Early stopping and outputs
    parser.add_argument("--enable_early_stop", type=str2bool, default=True)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args(argv)
    named_presets = sum(
        value is not None
        for value in (
            args.rq2_1_variant,
            args.rq3_variant,
            args.rq4_variant,
            args.rq5_mode,
            args.rq6_axis,
        )
    )
    if named_presets > 1:
        parser.error(
            "rq2_1_variant, rq3_variant, rq4_variant, rq5_mode, and rq6_axis are mutually exclusive"
        )
    _apply_rq2_1_variant(args)
    _apply_rq3_variant(args)
    _apply_rq4_variant(args)
    _apply_rq5_mode(args)
    try:
        _apply_rq6_axis(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not hasattr(args, "static_graph_init_source"):
        args.static_graph_init_source = "not_applicable"
    # Expose the active task cardinality to server-side components without
    # changing the legacy four-task constants used for metric naming.
    args.active_task_count = 1 if args.task_mode == "single" else 4
    if args.devices == "cuda" and not torch.cuda.is_available():
        args.devices = "cpu"
    if not 0 < args.train_density < 1:
        parser.error("train_density must be in (0, 1)")
    if not 0 <= args.valid_density < 1 - args.train_density:
        parser.error("valid_density leaves no test samples")
    if args.clients < 1 or not 0 < args.client_frac <= 1:
        parser.error("clients must be positive and client_frac must be in (0, 1]")
    if args.non_iid_alpha <= 0:
        parser.error("non_iid_alpha must be positive")
    if args.pfedgraph_alpha < 0 or args.pfedgraph_lambda < 0:
        parser.error("pFedGraph alpha and lambda must be non-negative")
    if args.pfedgraph_vector_size < 1:
        parser.error("pfedgraph_vector_size must be positive")
    if args.pfedln_neighbors < 1 or args.pfedln_temperature <= 0:
        parser.error("pFedLN neighbors and temperature must be positive")
    if not 0 <= args.pfedln_mix <= 1 or args.pfedln_vector_size < 1:
        parser.error("pFedLN mix must be in [0, 1] and vector size positive")
    if args.hcfgnn_neighbors < 1 or args.hcfgnn_attention_heads < 1:
        parser.error("HC-FGNN neighbors and attention heads must be positive")
    if args.hcfgnn_ldp_lambda < 0:
        parser.error("hcfgnn_ldp_lambda must be non-negative")
    if not 0 < args.flmfgm_upload_ratio <= 1:
        parser.error("flmfgm_upload_ratio must be in (0, 1]")
    if not 0 <= args.flmfgm_momentum < 1:
        parser.error("flmfgm_momentum must be in [0, 1)")
    if not args.fncf_hidden_dims or min(args.fncf_hidden_dims) < 1:
        parser.error("fncf_hidden_dims must contain positive integers")
    if not 0 <= args.graph_prior_weight <= 1:
        parser.error("graph_prior_weight must be in [0, 1]")
    if args.rq2_1_variant is not None and args.agg != "fedgl":
        parser.error("rq2_1_variant requires --agg fedgl")
    if args.rq3_variant is not None and args.agg != "fedgl":
        parser.error("rq3_variant requires --agg fedgl")
    if args.rq4_variant is not None and args.agg != "fedgl":
        parser.error("rq4_variant requires --agg fedgl")
    if args.rq6_axis is not None and args.agg != "fedgl":
        parser.error("rq6_axis requires --agg fedgl")
    if args.rq5_mode is not None:
        allowed = RQ5_METHODS_BY_MODE[args.rq5_mode]
        if args.agg not in allowed:
            parser.error(
                f"{args.rq5_mode} RQ5 mode supports methods: {', '.join(allowed)}"
            )
        if args.rq5_mode == "scalability" and args.clients not in {20, 50, 100}:
            parser.error("RQ5 scalability requires --clients to be one of 20, 50, or 100")
    if args.rq5_warmup_rounds < 0 or args.rq5_timed_rounds < 0:
        parser.error("rq5 warmup and timed rounds must be non-negative")
    if args.rq5_mode is not None and (
        args.rq5_warmup_rounds + args.rq5_timed_rounds > args.com_round
    ):
        parser.error("rq5 warmup_rounds + timed_rounds must not exceed com_round")
    if args.freeze_task_graphs and args.agg != "fedgl":
        parser.error("freeze_task_graphs requires --agg fedgl")
    if not 0 <= args.serverbeta <= 1:
        parser.error("serverbeta must be in [0, 1]")
    if args.dimension < 1 or args.heads < 1:
        parser.error("dimension and heads must be positive")
    if sum(args.task_weights) <= 0 or min(args.task_weights) < 0:
        parser.error("task_weights must be non-negative with a positive sum")
    return args
