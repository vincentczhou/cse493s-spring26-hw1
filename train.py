from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from model import GPT, GPTConfig
from utils import TextDataset, Tokenizer, load_data


@hydra.main(config_path="conf", version_base=None)
def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.general.seed)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name or cfg.general.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=Path(HydraConfig.get().runtime.output_dir),  # Hydra Output Dir
    )

    gptconfig = GPTConfig(**cfg.gptconfig)

    # Load data
    train_data = load_data(cfg.data.train_data_path)
    val_data = load_data(cfg.data.val_data_path)
    tokenizer = Tokenizer(train_data, gptconfig.vocab_size)
    gptconfig.vocab_size = len(tokenizer)
    train_ds = TextDataset(
        train_data, tokenizer, last_token_only=cfg.training.last_token_only
    )
    val_ds = TextDataset(
        val_data, tokenizer, last_token_only=cfg.training.last_token_only
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=TextDataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=TextDataset.collate_fn,
    )

    # Initialize model
    model = GPT(gptconfig).to(cfg.general.device)
    optimizer = model.configure_optimizers(
        cfg.training.weight_decay,
        cfg.training.lr,
        cfg.training.betas,
        cfg.general.device,
    )

    # Training loop
    step = 0
    pbar = tqdm(total=cfg.training.max_steps, desc="training")
    while step < cfg.training.max_steps:
        for x, y in train_loader:
            if step >= cfg.training.max_steps:
                break
            model.train()
            x, y = x.to(cfg.general.device), y.to(cfg.general.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, gptconfig.vocab_size), y.view(-1), ignore_index=-100
            )
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)

            if step % cfg.training.log_interval == 0:
                preds = logits.argmax(dim=-1)
                mask = y != -100
                train_acc = (preds[mask] == y[mask]).float().mean().item()
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{train_acc:.4f}"}
                )
                # Reporting train/loss and train/acc per STEP, not per epoch
                wandb.log(
                    {"train/loss": loss.item(), "train/acc": train_acc}, step=step
                )
            if step % cfg.training.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    for vx, vy in val_loader:
                        vx, vy = vx.to(cfg.general.device), vy.to(cfg.general.device)
                        vlogits = model(vx)
                        val_loss += F.cross_entropy(
                            vlogits.view(-1, gptconfig.vocab_size),
                            vy.view(-1),
                            ignore_index=-100,
                        ).item()

                        vpreds = vlogits.argmax(dim=-1)
                        vmask = vy != -100
                        val_correct += (vpreds[vmask] == vy[vmask]).sum().item()
                        val_total += vmask.sum().item()
                    val_loss = val_loss / len(val_loader)
                    val_acc = val_correct / val_total
                # Reporting val/loss and val/acc per val epoch, not per step
                wandb.log({"val/loss": val_loss, "val/acc": val_acc}, step=step)

            if step % cfg.training.save_interval == 0:
                ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "tokenizer": tokenizer,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "step": step,
                    },
                    ckpt_dir / f"ckpt_{step}.pt",
                )
    pbar.close()

    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "tokenizer": tokenizer,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "step": step,
        },
        ckpt_dir / "ckpt_final.pt",
    )


if __name__ == "__main__":
    train()
