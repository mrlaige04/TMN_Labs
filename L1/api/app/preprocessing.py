from io import BytesIO

import numpy as np
from PIL import Image

CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - CIFAR100_MEAN) / CIFAR100_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)
