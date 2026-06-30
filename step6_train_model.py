import os
import numpy as np
from skimage.io import imread
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib

train_dir = "train_data"

features = []
labels = []

characters = sorted(os.listdir(train_dir))
characters = [c for c in characters if os.path.isdir(os.path.join(train_dir, c))]

for char in characters:
    char_dir = os.path.join(train_dir, char)
    for filename in os.listdir(char_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(char_dir, filename)
            img = imread(img_path, as_gray=True)
            flattened = img.flatten()
            features.append(flattened)
            labels.append(char)

features = np.array(features)
labels = np.array(labels)

print("Total training samples: " + str(len(features)))
print("Feature vector size per sample: " + str(features.shape[1]))
print("Unique characters: " + str(len(set(labels))))

model = SVC(kernel="linear", probability=True, C=1.0)

print("Running 3-fold cross-validation to check accuracy...")
scores = cross_val_score(model, features, labels, cv=3)
print("Cross-validation scores: " + str(scores))
print("Average accuracy: " + str(scores.mean() * 100) + "%")

print("Training final model on all data...")
model.fit(features, labels)

joblib.dump(model, "lpr_model.pkl")
print("Model saved to lpr_model.pkl")
