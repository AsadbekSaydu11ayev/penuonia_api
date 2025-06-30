from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
# rasmlar bilan ishlash uchun
# working with images
from PIL import Image
# for reading images as bytes
# tasvirlarni bayt oqimi sifatida o'qish uchun
import io

# FastAPI ilovasini yaratamiz
# we will create FastAPI application
app = FastAPI() # you can lounch API via this app

# we upload the fastapi model
learn = load_learner("pnevmaniya_model.pkl")


@app.post("/predict")
async def predict_img(file: UploadFile = File(...)):

    contents = await file.read()
    # file bytes holatida o'qish

    img = PILImage.create(io.BytesIO(contents))

    pred, pred_id, probs = learn.predict(img)

    return {
        "prediction": str(pred), # Bashorat natijasini JSON formatda qaytaramiz
        "probability": f"{probs[pred_id]}"
    }
