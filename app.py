from fastapi import FastAPI, File,  UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from flask import Flask, redirect, url_for, request, render_template
from fastapi import FastAPI, File, UploadFile, Request

import requests
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import InputLayer
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.layers import TFSMLayer
from tensorflow.keras.models import load_model



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

#Loaded Model for Feature Extraction from Image
vgg16 = VGG16(weights='imagenet', include_top=True)
fc2_layer = vgg16.get_layer('fc2').output
model_fc2 = tf.keras.Model(inputs=vgg16.input, outputs=fc2_layer)

Model = tf.keras.models.load_model("caption_generator_model.h5")

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 50

#preproecessing Text
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = Model.predict([tf.convert_to_tensor(image), tf.convert_to_tensor(sequence)])
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

#Convert Index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

#Reading file
def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = preprocess_input(img)
    feature = model_fc2.predict(img, verbose=0)
    return feature

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    input_image_feature = read_file_as_image(await file.read())
    prediction = predict_caption(Model, input_image_feature, tokenizer, max_length)
    return {"caption": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

