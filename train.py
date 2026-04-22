import argparse

import torch

from utils import TextDataset, load_config, load_data, train_word_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load configuration
    config_dict, gptconfig = load_config(args.config)

    # Load data
    data = load_data(config_dict["general"]["train_data_path"])
    word_tokenizer = train_word_tokenizer(data, gptconfig.vocab_size)
    train_ds = TextDataset(data, word_tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config_dict["general"]["batch_size"],
        shuffle=True,
        collate_fn=TextDataset.collate_fn,
    )

    import pdb

    pdb.set_trace()
    # Train and log and do stuff here


if __name__ == "__main__":
    main()
