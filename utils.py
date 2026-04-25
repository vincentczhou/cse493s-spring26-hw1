from pathlib import Path

import torch
from torch.utils.data import Dataset

UNK = "<UNK>"
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3


def load_data(data_path: Path) -> list[str]:
    data = []
    with open(data_path, "r") as f:
        data = [line.strip() for line in f]
    return data


class Tokenizer:
    def __init__(self, data: list[str], vocab_size: int):
        # Why word level tokenizer? Because that's what the grokking paper does!
        # Create a set of unique tokens
        unique_tokens = set()
        for line in data:
            tokens = line.split()  # Split by whitespace
            unique_tokens.update(tokens)

        # TODO: Limit the vocabulary size, this should be done more intelligently in practice (e.g., by frequency), but for simplicity, we'll just take the first `vocab_size` tokens because for the task datasets our vocab size will never be that big.
        # It's important that the vocab size specified in the config is larger than the actual number of unique tokens in the data (given our task), otherwise we might end up with a smaller vocab size than expected after truncation, which can cause issues during training.
        if len(unique_tokens) > vocab_size:
            unique_tokens = set(list(unique_tokens)[:vocab_size])

        # Create a token to index mapping
        # Start indexing from 4 to reserve 0-3 for special tokens
        self.token_to_index = {
            token: idx for idx, token in enumerate(unique_tokens, start=4)
        }
        self.token_to_index[UNK] = UNK_IDX
        self.token_to_index[PAD] = PAD_IDX
        self.token_to_index[BOS] = BOS_IDX
        self.token_to_index[EOS] = EOS_IDX
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def encode(self, s: str) -> list[int]:
        return (
            [BOS_IDX]
            + [self.token_to_index.get(t, UNK_IDX) for t in s.split()]
            + [EOS_IDX]
        )

    def decode(self, indices: list[int]) -> str:
        # skip = {BOS_IDX, EOS_IDX, PAD_IDX}
        # return ' '.join(self.index_to_token[i] for i in indices if i not in skip)
        return " ".join(self.index_to_token[i] for i in indices)

    def __len__(self):
        return len(self.token_to_index)


class TextDataset(Dataset):
    def __init__(
        self, samples: list[str], tokenizer: Tokenizer, last_token_only: bool = False
    ):
        self.tokenized_data = [
            torch.tensor(tokenizer.encode(s), dtype=torch.long) for s in samples
        ]
        self.last_token_only = last_token_only

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        tokens = self.tokenized_data[idx]
        x = tokens[:-1]
        y = tokens[1:].clone()
        if self.last_token_only:
            # the last token is EOS, the token of interest is the one before EOS
            y[:-2] = -100
            y[-1] = -100  # mask EOS
        return x, y

    def collate_fn(batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=True, padding_value=PAD_IDX
        )
        ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return xs, ys
