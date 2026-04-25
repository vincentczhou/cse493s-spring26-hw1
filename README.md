
# CSE 493s HW 1

Author: Vincent Chau

This file contains instructions on how to run the code, and expected outputs.

## Setup

Requires Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up pre-commit hooks
uv run pre-commit install

# Log in to WandB (one-time)
uv run wandb login
```

## Running Training

Training is configured via YAML files in `conf/` managed by `hydra`. Run with:

```bash
uv run train.py --config-name=<config>
```

For example, to run the sanity check:

```bash
uv run train.py --config-name=sanitycheck
```

Individual config values can be overridden from the command line:

```bash
uv run train.py --config-name=sanitycheck gptconfig.n_layer=2 training.lr=1e-3
```

Outputs (checkpoints, logs) are saved to `outputs/<run_name>/<timestamp>/`. Metrics are logged to WandB.

<!-- # Advanced ML HW 1

This is the starter code for CSE493S HW 1 for Spring 2026. For a complete description of the homework please see `ps.md` or `ps.pdf`. Please contact the TAs if you find any bugs in this repo. 

## Part 0 and 1

The file `model.py` contains an implementation of a transformer model. The file `part_0_1_contract.py` contains some function signatures that would make autograding less painful for the TAs. 

## Part 2

The notebook `part_2_starter.ipynb` has code to load pretrained models, the AIME dataset, functions for evaluation and has unoptimized code for inference. We encourage you to write your own inference code to speed things up. -->
