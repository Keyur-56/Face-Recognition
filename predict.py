import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json

from tensorflow.keras.models import load_model

# SETTINGS

IMG_SIZE = 128

people = []

with open("classes.json", "r") as f:
    people = json.load(f)

MODEL_PATH = "simpsons_model.keras"
IMAGE_PATH = r"simpsons_dataset\milhouse_van_houten\pic_0014.jpg"

# LOAD MODEL
print("Loading Model...")

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# LOAD IMAGE
import os

print("Current Working Directory:", os.getcwd())
print("Image Path:", IMAGE_PATH)
print("File Exists:", os.path.exists(IMAGE_PATH))

img = cv.imread(IMAGE_PATH)

if img is None:
    print("OpenCV could not read the image!")
    exit()

# PREPROCESS IMAGE
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = cv.resize(gray, (IMG_SIZE, IMG_SIZE))

gray = gray.astype("float32") / 255.0

gray = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# PREDICT
prediction = model.predict(gray, verbose=0)

index = np.argmax(prediction)

confidence = np.max(prediction)

print("\nPrediction :", people[index])

print(f"Confidence : {confidence*100:.2f}%")

# SHOW IMAGE
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(rgb)

plt.title(f"{people[index]} ({confidence*100:.2f}%)")

plt.axis("off")

plt.show()