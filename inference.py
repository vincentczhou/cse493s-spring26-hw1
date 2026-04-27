import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

from model import GPT, GPTConfig

log = logging.getLogger(__name__)


def load_checkpoint(checkpoint_dir: str, device: str = "cpu"):
    ckpt_dir = Path(checkpoint_dir)
    ckpt_path = ckpt_dir / "ckpt_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    tokenizer = ckpt["tokenizer"]
    cfg = ckpt["config"]
    gptconfig = GPTConfig(**cfg["gptconfig"])
    gptconfig.vocab_size = len(tokenizer)  # original hydra cfg is saved, not gptconfig
    model = GPT(gptconfig).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def generate_next_token(model, tokenizer, input_str: str) -> str:
    device = next(model.parameters()).device
    tokens = tokenizer.encode(input_str)[:-1]  # strip EOS added by encode
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(
        0
    )  # add batch dim
    with torch.no_grad():
        logits = model(x)
        next_token = logits[0, -1, :].argmax().item()
    return tokenizer.decode([next_token])


@hydra.main(
    config_path="conf/inference", config_name="0.1_sanitycheck_01", version_base=None
)
def main(cfg: DictConfig) -> None:
    model, tokenizer = load_checkpoint(cfg.checkpoint_dir, cfg.general.device)
    log.info("Starting Inference...")
    with open(cfg.input_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            output = generate_next_token(model, tokenizer, line)
            log.info(
                f"{i}: (Input) - {line} (Output) - {output} (Combined) - {line} {output}"
            )


if __name__ == "__main__":
    main()
