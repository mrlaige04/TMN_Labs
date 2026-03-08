import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import get_default_cfg
from src.data import build_loaders
from src.engine import run_epoch
from src.model import CustomResNet
from src.seed import set_seed
from src.visualize import plot_curves, show_predictions

def main():
    cfg = get_default_cfg()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Device:", device)

    train_loader, val_loader, classes = build_loaders(cfg)
    num_classes = len(classes)
    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Classes:", num_classes)

    model = CustomResNet(
        num_classes=num_classes,
        blocks_per_stage=(2, 2, 2),
        base_channels=64,
    ).to(device)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_path = "best_custom_resnet_cifar100.pt"

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_mode=True,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_mode=False,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    print(f"Finished in {(time.time() - start) / 60:.2f} min")
    print(f"Best val acc: {best_val_acc:.4f} | saved: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    plot_curves(history)
    show_predictions(model, val_loader, classes, device=device, n_show=8)


if __name__ == "__main__":
    main()
