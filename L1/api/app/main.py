import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from .onnx_predictor import ONNXPredictor
from .preprocessing import preprocess_image


MODEL_PATH = os.getenv("MODEL_PATH", "models/custom_resnet_cifar100.onnx")

app = FastAPI(title="CIFAR-100 Inference API", version="1.0.0")
predictor = ONNXPredictor(model_path=MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    input_tensor = preprocess_image(image_bytes)
    pred = predictor.predict(input_tensor=input_tensor, top_k=5)
    return pred
