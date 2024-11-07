from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import os
import requests

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

app = FastAPI()

#MODEL = keras.models.load_model("./models/potato_disease/1.keras")
endpoint = "http://localhost:8505/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    #predictions = MODEL.predict(image_batch)
    request_object = {
       "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=request_object)
    predictions = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(np.max(predictions) * 100, 2)
    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8080")
