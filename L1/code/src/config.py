from dataclasses import dataclass
@dataclass
class CFG:
    seed: int = 42
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    train_fraction: float = 0.6
    val_fraction: float = 0.6
def get_default_cfg() -> CFG:
    return CFG()
