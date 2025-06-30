from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from io import BytesIO
import pathlib, platform
plt = platform.system()
if plt == "Linux": pathlib.WindowsPath = pathlib.PosixPath

app = FastAPI()
learn = load_learner("export.pkl")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = PILImage.create(BytesIO(contents))
    pred_class, pred_idx, probs = learn.predict(img)
    return {
        "prediction": str(pred_class),
        "probability": f"{probs[pred_idx]:.4f}"
    }
