import sys
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import numpy as np
import joblib

image_path = sys.argv[1] if len(sys.argv) > 1 else "cropped_plate2.png"
plate_image = imread(image_path)

plate_float = img_as_float(plate_image)
gray_plate = rgb2gray(plate_float)
threshold_value = threshold_otsu(gray_plate)
binary_plate = gray_plate < threshold_value

label_image = measure.label(binary_plate)
plate_height, plate_width = gray_plate.shape

char_min_height = 0.35 * plate_height
char_max_height = 0.95 * plate_height
char_min_width = 0.03 * plate_width
char_max_width = 0.3 * plate_width

characters = []
column_positions = []

for region in regionprops(label_image):
    min_row, min_col, max_row, max_col = region.bbox
    h = max_row - min_row
    w = max_col - min_col
    if char_min_height <= h <= char_max_height and char_min_width <= w <= char_max_width:
        char_image = binary_plate[min_row:max_row, min_col:max_col]
        characters.append(char_image)
        column_positions.append(min_col)

print("Found " + str(len(characters)) + " character(s)")

sorted_chars = [c for _, c in sorted(zip(column_positions, characters), key=lambda pair: pair[0])]

model = joblib.load("lpr_model.pkl")
predicted_plate = ""
for char in sorted_chars:
    resized = resize(char, (20, 20))
    binary = (resized * 255).astype(np.uint8)
    flattened = binary.flatten().reshape(1, -1)
    prediction = model.predict(flattened)[0]
    predicted_plate += prediction

print("Predicted plate number: " + predicted_plate)
