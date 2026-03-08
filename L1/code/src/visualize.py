import matplotlib.pyplot as plt
import torch

from .data import MEAN, STD


def plot_curves(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_predictions(model, val_loader, classes, device, n_show: int = 8):
    model.eval()

    inv_mean = torch.tensor(MEAN).view(3, 1, 1)
    inv_std = torch.tensor(STD).view(3, 1, 1)

    def denorm(img):
        return torch.clamp(img * inv_std + inv_mean, 0, 1)

    x_batch, y_batch = next(iter(val_loader))
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)
        preds = logits.argmax(dim=1)

    plt.figure(figsize=(14, 4))
    for i in range(n_show):
        plt.subplot(2, 4, i + 1)
        img = denorm(x_batch[i].cpu()).permute(1, 2, 0).numpy()
        plt.imshow(img)
        true_label = classes[y_batch[i].item()]
        pred_label = classes[preds[i].item()]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
