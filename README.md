# FedGL

Core implementation of **FedGL: Task-Specific Graphs for Layer-Wise Federated Aggregation in Blockchain Service Reliability Prediction**.

FedGL combines client-local context and interaction modeling, asymmetric multi-task prediction, task-specific client collaboration graphs, graph-guided layer-wise aggregation, and personalized residual fusion. It predicts success rate (SR), right block ratio (RB), recent height ratio (RH), and round-trip time (RTT).

## Repository contents

```text
code/
  main.py                 Training entry point
  config.py               Configuration and experiment presets
  requirements.txt        Python dependencies
  core_utils/             Metrics and state utilities
  data_processing/        QoS loading and client partitioning
  graph_construction/     Context graphs and client prior graph
  models/                 Local predictor
  federated/              Local training, server, and aggregation
  baseline/               Baseline adapters used by the training code
```

This repository contains source code and its required baseline modules. Datasets, paper files, plotting scripts, checkpoints, experiment outputs, and local test utilities are excluded.

## Environment

The locally validated GPU environment uses Python 3.10, PyTorch 2.1.2, CUDA 12.1, and DGL 2.2.1 with CUDA 12.1 support. Install compatible PyTorch and DGL builds for your platform; `requirements.txt` records package constraints, but does not select CUDA-specific wheels.

```bash
pip install -r code/requirements.txt
```

## Data preparation

The experiments use the H-BRP dataset introduced by Peilin Zheng, Zibin Zheng, and Liang Chen in *Selecting reliable blockchain peers via hybrid blockchain reliability prediction*, IET Software (2023), DOI: [10.1049/sfw2.12118](https://doi.org/10.1049/sfw2.12118). Refer to that study for the dataset and its access conditions. Data are not redistributed here.

Prepare a directory containing `ClientWithCTX.csv`, `PeerWithCTX.csv`, and the four processed QoS matrices for the selected setting. For the default suffix `_12_1000`, these are:

```text
ClientWithCTX.csv
PeerWithCTX.csv
SuccessRate_12_1000.csv
rightBlock_12_1000.csv
recentHeight_12_1000.csv
roundtripTime_12_1000.csv
```

Context files are comma-separated tables containing the configured attributes (by default `as`, `country`, `isp`, and `timezone`). QoS files are tab-separated numeric matrices without a header, with identical user/service ordering across tasks and context tables. Each matrix has one row per user and one column per service. Zero is a valid observed value.

## Run

```bash
python code/main.py --agg fedgl --rq2_1_variant full --dataset_path /absolute/path/to/data
```

For a short CPU smoke run:

```bash
python code/main.py --devices cpu --agg fedgl --dataset_path /absolute/path/to/data --clients 4 --client_frac 1 --train_density 0.01 --valid_density 0.01 --com_round 1 --client_epochs 1 --graph_evolution_epochs 1 --save_model false
```

Relative dataset and output paths are resolved from the `code/` directory. Training history, final test metrics, and configuration are written to the configured `--output_dir`. Use `python code/main.py --help` to inspect the available options. Reproducing a particular experiment requires its explicit configuration rather than relying on defaults.
