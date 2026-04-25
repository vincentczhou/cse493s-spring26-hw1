import random
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm


def generate(p: int, operator: str) -> list[str]:
    data = []
    if operator == "+":
        for a in range(p + 1):
            for b in range(p + 1):
                data.append(f"{a} + {b} = {(a + b) % p}")
    elif operator == "-":
        for a in range(p + 1):
            for b in range(p + 1):
                data.append(f"{a} - {b} = {(a - b) % p}")
    elif operator == "/":
        for a in range(p + 1):
            for b in range(1, p):
                c = (a * pow(b, -1, p)) % p
                data.append(f"{a} / {b} = {c}")
    return data


def split_and_save(data: list[str], output_dir: Path, name: str, train_frac: float):
    train_size = int(len(data) * train_frac)
    train, test = data[:train_size], data[train_size:]
    (output_dir / f"{name}_train.txt").write_text("\n".join(train))
    (output_dir / f"{name}_test.txt").write_text("\n".join(test))
    print(f"{name}: {len(train)} train, {len(test)} test")


@hydra.main(config_path="conf/data", config_name="1.1_generate_data", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)

    op_names = {"+": "add", "-": "sub", "/": "div"}
    combos = [(p, op) for p in cfg.ps for op in cfg.operators]
    for p, op in tqdm(combos, desc="generating"):
        data = generate(p, op)
        random.shuffle(data)
        split_and_save(data, output_dir, f"{op_names[op]}_p{p}", cfg.train_frac)


if __name__ == "__main__":
    main()
