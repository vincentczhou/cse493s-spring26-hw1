from pathlib import Path

import torch
from torch.utils.data import Dataset
# import yaml


UNK = "<UNK>"
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


# def load_config(config_path: Path) -> tuple[dict, GPTConfig]:
#     c = GPTConfig()
#     with open(config_path, "r") as f:
#         config_dict = yaml.safe_load(f)
#     for key, value in config_dict["gptconfig"].items():
#         setattr(c, key, value)
#     return config_dict, c


def load_data(data_path: Path) -> list[str]:
    data = []
    with open(data_path, "r") as f:
        data = [line.strip() for line in f]
    return data


def train_word_tokenizer(data: list[str], vocab_size: int) -> dict[str, int]:
    # Why word level tokenizer? Because that's what the grokking paper does!
    # Create a set of unique tokens
    unique_tokens = set()
    for line in data:
        tokens = line.split()  # Split by whitespace
        unique_tokens.update(tokens)

    # TODO: Limit the vocabulary size, this should be done more intelligently in practice (e.g., by frequency), but for simplicity, we'll just take the first `vocab_size` tokens because for the task datasets our vocab size will never be that big.
    if len(unique_tokens) > vocab_size:
        unique_tokens = set(list(unique_tokens)[:vocab_size])

    # Create a token to index mapping
    token_to_index = {
        token: idx for idx, token in enumerate(unique_tokens, start=4)
    }  # Start indexing from 4 to reserve 0-3 for special tokens
    token_to_index[UNK] = UNK_IDX
    token_to_index[PAD] = PAD_IDX
    token_to_index[BOS] = BOS_IDX
    token_to_index[EOS] = EOS_IDX

    return token_to_index


def tokenize(data: list[str], token_to_index: dict[str, int]) -> list[int]:
    tokenized_data = []
    for line in data:
        tokens = line.split()  # Split by whitespace
        token_indices = [
            token_to_index.get(token, token_to_index[UNK]) for token in tokens
        ]
        token_indices = [token_to_index[BOS]] + token_indices + [token_to_index[EOS]]
        tokenized_data.append(token_indices)
    return tokenized_data


class TextDataset(Dataset):
    def __init__(self, samples: list[str], token_to_index: dict[str, int]):
        self.tokenized_data = torch.tensor(tokenize(samples, token_to_index))

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        return tokens[:-1], tokens[
            1:
        ]  # Input sequence and target sequence (shifted by one)

    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=True, padding_value=PAD_IDX
        )
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return xs, ys
