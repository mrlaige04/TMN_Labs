from typing import Tuple

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    train_mode: bool = True,
) -> Tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)

            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item()
        running_acc += accuracy(logits, y)
        n_batches += 1

    return running_loss / n_batches, running_acc / n_batches
