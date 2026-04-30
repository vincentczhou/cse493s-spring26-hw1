# CSE 493s HW 1

Author: Vincent Chau

This file contains instructions on how to run the code, the environment definition, and notes for the autograder.

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
├── train.py                    # main training script
├── inference.py                # inference / generation script
├── generate_grok_data.py       # modular arithmetic data generation
├── model.py                    # nanoGPT model (unchanged from starter)
├── utils.py                    # Tokenizer + TextDataset
└── part_0_1_contract.py        # autograder interface
```

## Generating data

Modular arithmetic data for section 1 is generated with:

```bash
uv run generate_grok_data.py --config-name=1.1_generate_data
```

Configurable via `conf/data/*.yaml`

## Running training

Training is configured via `conf/experiment/*.yaml` managed by Hydra:

```bash
uv run train.py --config-name=<config>
```

Examples:

```bash
# Sanity check (Section 0.1)
uv run train.py --config-name=0.1_sanitycheck_01
```

Outputs (checkpoints, logs, configs) are saved under `outputs/<run_name>/<timestamp>/`. Metrics stream to WandB.

## Running inference

Inference uses Hydra configs in `conf/inference/`:

```bash
uv run inference.py --config-name=0.1_sanitycheck_01
```

The script reads one input per line and prints the model's next-token prediction (single-token greedy decoding).

## Autograder notes (`part_0_1_contract.py`)

The contract functions are implemented in `part_0_1_contract.py` and import from `inference.py` and `utils.py`.

Following the grokking paper, separate models are trained for each `(operator, p)` pair - `p` is not encoded in the input sequence, and all inputs have the form `"a ◦ b = c"` (`◦` is replaced by the actual operator character). Note that:

- `predict_answer(model, tokenizer, a, b, op, p)` **ignores the `p` argument**. The correct `p` is implicit in the loaded model, so callers should ensure `load_model_and_tokenizer` was given the checkpoint trained for that `(op, p)` pair before calling `predict_answer`.
- Calling `predict_answer` with an operator whose token wasn't seen during training would produce an `<UNK>` and a wrong prediction - the operator must match the model's training operation.

**`load_model_and_tokenizer(checkpoint_dir)`** expects a directory containing a `ckpt_final.pt` file (produced by `train.py` at the end of training).

**`get_bos_token()`** returns the literal string `"<BOS>"` used by the word-level tokenizer.
<!-- # Advanced ML HW 1

This is the starter code for CSE493S HW 1 for Spring 2026. For a complete description of the homework please see `ps.md` or `ps.pdf`. Please contact the TAs if you find any bugs in this repo. 

## Part 0 and 1

The file `model.py` contains an implementation of a transformer model. The file `part_0_1_contract.py` contains some function signatures that would make autograding less painful for the TAs. 

## Part 2

The notebook `part_2_starter.ipynb` has code to load pretrained models, the AIME dataset, functions for evaluation and has unoptimized code for inference. We encourage you to write your own inference code to speed things up. -->