from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CFG

MEAN: Tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
STD: Tuple[float, float, float] = (0.2675, 0.2565, 0.2761)

def _make_subset(ds, fraction: float, seed: int):
    if fraction >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(ds, idx)

def build_loaders(cfg: CFG):
    train_tfms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    full_train = datasets.CIFAR100(
        root=cfg.data_dir, train=True, download=True, transform=train_tfms
    )
    full_val = datasets.CIFAR100(
        root=cfg.data_dir, train=False, download=True, transform=val_tfms
    )

    train_ds = _make_subset(full_train, cfg.train_fraction, cfg.seed)
    val_ds = _make_subset(full_val, cfg.val_fraction, cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    classes: List[str] = full_train.classes
    return train_loader, val_loader, classes
