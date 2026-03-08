from pathlib import Path

import torch

from app.torch_model import CustomResNet


def export_onnx(
    checkpoint_path: str = "../code/best_custom_resnet_cifar100.pt",
    onnx_path: str = "models/custom_resnet_cifar100.onnx",
):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. "
            "Train the model first or provide a correct path."
        )

    onnx_file = Path(onnx_path)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    model = CustomResNet(num_classes=100, blocks_per_stage=(2, 2, 2), base_channels=64)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_file),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"ONNX model exported to: {onnx_file}")


if __name__ == "__main__":
    export_onnx()
