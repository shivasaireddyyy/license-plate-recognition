import os
import numpy as np
from skimage.io import imread
import joblib

model = joblib.load("lpr_model.pkl")

char_dir = "segmented_characters"
filenames = sorted(os.listdir(char_dir), key=lambda x: int(x.split("_")[1].split(".")[0]))

predicted_plate = ""

for filename in filenames:
    if filename.endswith(".png"):
        img_path = os.path.join(char_dir, filename)
        img = imread(img_path, as_gray=True)
        flattened = img.flatten().reshape(1, -1)
        prediction = model.predict(flattened)[0]
        predicted_plate += prediction
        print(filename + " -> predicted: " + prediction)

print("")
print("Predicted plate number: " + predicted_plate)
