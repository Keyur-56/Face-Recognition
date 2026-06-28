import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128

DATASET = "simpsons_dataset"

folder_counts = {}

for folder in os.listdir(DATASET):

    folder_path = os.path.join(DATASET, folder)

    if not os.path.isdir(folder_path):
        continue

    count = len([
        file
        for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    folder_counts[folder] = count

folder_counts = sorted(
    folder_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

people = [folder for folder, count in folder_counts[:10]]

print(people)

print()

for folder, count in folder_counts[:10]:
    print(folder, count)

features = []
labels = []

for label, person in enumerate(people):

    folder = os.path.join(DATASET, person)

    print("Loading", person)

    for img in os.listdir(folder):

        if not img.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, img)

        image = cv.imread(path)

        if image is None:
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        gray = cv.resize(gray, (IMG_SIZE, IMG_SIZE))

        features.append(gray)

        labels.append(label)

print("Images :", len(features))
print("Labels :", len(labels))

X = np.array(features, dtype="float32")

X = X / 255.0

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = np.array(labels)

y = to_categorical(y, len(people))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(IMG_SIZE,IMG_SIZE,1)
    )
)

model.add(MaxPooling2D())

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())

model.add(
    Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())

model.add(
    Conv2D(
        256,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(len(people), activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=10,
    batch_size=32
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(accuracy*100)

with open("classes.json", "w") as f:
    json.dump(people, f)

model.save("simpsons_model.keras")

print(os.path.exists("simpsons_model.keras"))