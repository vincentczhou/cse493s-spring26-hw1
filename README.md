# CSE 493s HW 1

Author: Vincent Chau

This file contains instructions on how to run the code, the environment definition, and notes for the autograder. Please refer to `report.pdf` for our writeup.

## Table of Contents

- [CSE 493s HW 1](#cse-493s-hw-1)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Project layout](#project-layout)
  - [Part 0: Train Infrastructure Setup](#part-0-train-infrastructure-setup)
    - [0.1 Sanity Checks](#01-sanity-checks)
  - [Part 1: Training on Algorithmic Tasks](#part-1-training-on-algorithmic-tasks)
    - [1.1 Data Generation](#11-data-generation)
    - [1.2 Warmup — Addition and Subtraction Experiments](#12-warmup--addition-and-subtraction-experiments)
    - [1.3 Grokking](#13-grokking)
    - [1.4 Analysis](#14-analysis)
    - [Autograder notes](#autograder-notes)
  - [Part 2: Test-Time Scaling](#part-2-test-time-scaling)
  - [Part 3: Synthesis and Analysis](#part-3-synthesis-and-analysis)

## Setup

Requires Python 3.12+. Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up pre-commit hooks (optional, dev only)
uv run pre-commit install

# Log in to WandB (one-time)
uv run wandb login
```

Versions and dependencies are pinned in `pyproject.toml` and resolved via `uv.lock`.

## Project layout

```
.
├── conf/                       # Hydra configs
│   ├── data/                   # data generation configs
│   ├── experiment/             # training configs (sanity check, 1.2, 1.3, 1.4)
│   └── inference/              # inference configs
├── data/                       # generated/static datasets
├── outputs/                    # Hydra run outputs (checkpoints, logs)
├── results/                    # Part 2 inference result CSVs
├── train.py                    # main training script (Parts 0, 1)
├── inference.py                # inference / generation script (Parts 0, 1)
├── generate_grok_data.py       # modular arithmetic data generation (Part 1)
├── model.py                    # nanoGPT model (unchanged from starter)
├── utils.py                    # Tokenizer + TextDataset
├── part_0_1_contract.py        # autograder interface (Parts 0, 1)
└── part_2.ipynb                # Part 2 notebook (AIME-2024 inference + scaling)
```

## Part 0: Train Infrastructure Setup

Part 0 establishes the training/inference infrastructure: `train.py`, `inference.py`, `model.py`, `utils.py`. Outputs (checkpoints, logs, configs) are saved under `outputs/<run_name>/<timestamp>/`; metrics stream to WandB.

### 0.1 Sanity Checks

Sanity-check training and inference use `conf/experiment/0.1_sanitycheck_*.yaml`:

```bash
# Train sanity-check model
uv run train.py --config-name=0.1_sanitycheck_01

# Run inference with sanity-check config
uv run inference.py --config-name=0.1_sanitycheck_01
```

The inference script reads one input per line and prints the model's next-token prediction (single-token greedy decoding).

## Part 1: Training on Algorithmic Tasks

Part 1 trains small transformers on modular arithmetic to study grokking, using the same `train.py` / `inference.py` from Part 0 and the configs under `conf/experiment/1.*.yaml`.

Following the grokking paper, separate models are trained for each `(operator, p)` pair — `p` is not encoded in the input sequence, and all inputs have the form `"a ◦ b = c"` (`◦` is replaced by the actual operator character).

For training and inference in any of 1.2 / 1.3 / 1.4:

```bash
uv run train.py --config-name=<config>       # configs under conf/experiment/
uv run inference.py --config-name=<config>   # configs under conf/inference/
```

### 1.1 Data Generation

```bash
uv run generate_grok_data.py --config-name=1.1_generate_data
```

Configurable via `conf/data/*.yaml`.

### 1.2 Warmup — Addition and Subtraction Experiments

Trains 1- and 2-layer transformers on modular addition and subtraction for `p ∈ {97, 113}`. Configs follow the pattern `1.2_<op>_p<p>_<layers>_<seed>.yaml` (e.g. `1.2_add_p97_1_493`, `1.2_sub_p113_2_493`).

### 1.3 Grokking

Reproduces Fig. 1 of the grokking paper on modular division for `p = 97`. Configs: `1.3_div_p97_*_493.yaml`.

### 1.4 Analysis

Ablations on the modular division task varying training fraction (`1.4_div_train_*`) and weight decay (`1.4_div_wd_*`) to measure their effect on time-to-grok.

### Autograder notes

The contract functions are implemented in `part_0_1_contract.py` and import from `inference.py` and `utils.py`.

- `predict_answer(model, tokenizer, a, b, op, p)` ignores the `p` argument. The correct `p` is implicit in the loaded model, so callers should ensure `load_model_and_tokenizer` was given the checkpoint trained for that `(op, p)` pair before calling `predict_answer`.
- Calling `predict_answer` with an operator whose token wasn't seen during training would produce an `<UNK>` and a wrong prediction — the operator must match the model's training operation.

`load_model_and_tokenizer(checkpoint_dir)` expects a directory containing a `ckpt_final.pt` file (produced by `train.py` at the end of training).

`get_bos_token()` returns the literal string `"<BOS>"` used by the word-level tokenizer.

## Part 2: Test-Time Scaling

Part 2 lives in `part_2.ipynb`. It loads `Qwen/Qwen3-4B` via vLLM and the `OpenRLHF/aime-2024` dataset. Inference results are written as CSVs under `results/` and re-loaded by the plotting cells, so each section can be re-run independently without re-doing inference. Open the notebook and run cells top-to-bottom. See notebook for more details.

## Part 3: Synthesis and Analysis

Please refer to the writeup for more details.

<!-- # Advanced ML HW 1

This is the starter code for CSE493S HW 1 for Spring 2026. For a complete description of the homework please see `ps.md` or `ps.pdf`. Please contact the TAs if you find any bugs in this repo. 

## Part 0 and 1

The file `model.py` contains an implementation of a transformer model. The file `part_0_1_contract.py` contains some function signatures that would make autograding less painful for the TAs. 

## Part 2

The notebook `part_2_starter.ipynb` has code to load pretrained models, the AIME dataset, functions for evaluation and has unoptimized code for inference. We encourage you to write your own inference code to speed things up. -->
