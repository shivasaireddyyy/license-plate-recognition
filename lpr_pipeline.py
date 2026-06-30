import sys
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import joblib


def detect_plate(car_image):
    car_image_float = img_as_float(car_image)
    gray_car_image = rgb2gray(car_image_float)
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    label_image = measure.label(binary_car_image)

    min_height, max_height = 40, 90
    min_width, max_width = 200, 350

    best_region = None
    best_area = 0

    for region in regionprops(label_image):
        if region.area < 50:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        h = max_row - min_row
        w = max_col - min_col

        if (min_height <= h <= max_height and
                min_width <= w <= max_width and
                w > h):
            if region.area > best_area:
                best_area = region.area
                best_region = (min_row, min_col, max_row, max_col)

    if best_region is None:
        return None

    min_row, min_col, max_row, max_col = best_region
    margin = 2
    plate = car_image[max(0, min_row - margin):max_row + margin,
                       max(0, min_col - margin):max_col + margin]
    return plate


def segment_characters(plate_image):
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

    sorted_chars = [c for _, c in sorted(zip(column_positions, characters), key=lambda pair: pair[0])]
    return sorted_chars


def predict_characters(characters, model):
    predicted_plate = ""
    for char in characters:
        resized = resize(char, (20, 20))
        binary = (resized * 255).astype(np.uint8)
        flattened = binary.flatten().reshape(1, -1)
        prediction = model.predict(flattened)[0]
        predicted_plate += prediction
    return predicted_plate


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 lpr_pipeline.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("Error: file not found -> " + image_path)
        sys.exit(1)

    if not os.path.exists("lpr_model.pkl"):
        print("Error: lpr_model.pkl not found. Train the model first.")
        sys.exit(1)

    print("Loading image: " + image_path)
    car_image = imread(image_path)

    print("Detecting license plate...")
    plate = detect_plate(car_image)

    if plate is None:
        print("Could not detect a license plate in this image.")
        print("You may need to adjust the size thresholds in detect_plate() for this image.")
        sys.exit(1)

    imsave("last_detected_plate.png", plate)
    print("Plate detected and saved to last_detected_plate.png")

    print("Segmenting characters...")
    characters = segment_characters(plate)
    print("Found " + str(len(characters)) + " character(s)")

    if len(characters) == 0:
        print("No characters could be segmented from the plate.")
        sys.exit(1)

    print("Loading recognition model...")
    model = joblib.load("lpr_model.pkl")

    print("Recognizing characters...")
    plate_number = predict_characters(characters, model)

    print("")
    print("=" * 40)
    print("PREDICTED PLATE NUMBER: " + plate_number)
    print("=" * 40)


if __name__ == "__main__":
    main()
    