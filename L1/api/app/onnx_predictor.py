from pathlib import Path

import numpy as np
import onnxruntime as ort

from .labels import CIFAR100_CLASSES


class ONNXPredictor:
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"ONNX model not found at '{model_file}'. "
                "Run export_to_onnx.py first."
            )

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_file), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray, top_k: int = 5):
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        logits = logits[0]

        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        top_idx = np.argsort(-probs)[:top_k]
        top = [
            {
                "class_id": int(i),
                "label": CIFAR100_CLASSES[int(i)],
                "probability": float(probs[int(i)]),
            }
            for i in top_idx
        ]

        return {
            "top1": top[0],
            "top_k": top,
        }
