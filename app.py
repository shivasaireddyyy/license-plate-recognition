import streamlit as st
import numpy as np
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import joblib
import os

st.set_page_config(page_title="License Plate Recognition", page_icon="🚗", layout="centered")


def detect_plate(car_image):
    car_image_float = img_as_float(car_image)
    gray_car_image = rgb2gray(car_image_float)
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    label_image = measure.label(binary_car_image)

    h_img, w_img = label_image.shape

    # Broader, more forgiving size range (as a fraction of image dimensions)
    min_height, max_height = 0.01 * h_img, 0.20 * h_img
    min_width, max_width = 0.05 * w_img, 0.55 * w_img

    candidates = []

    for region in regionprops(label_image):
        if region.area < 50:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        h = max_row - min_row
        w = max_col - min_col

        if h == 0:
            continue
        aspect_ratio = w / h

        # Plates are wider than tall, roughly 2:1 to 6:1 ratio
        if (min_height <= h <= max_height and
                min_width <= w <= max_width and
                2.0 <= aspect_ratio <= 6.5):
            candidates.append((region.area, min_row, min_col, max_row, max_col))

    if not candidates:
        return None

    # Pick the largest matching candidate (most likely to be the real plate)
    candidates.sort(reverse=True, key=lambda c: c[0])
    _, min_row, min_col, max_row, max_col = candidates[0]

    margin = 3
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


st.title("🚗 License Plate Recognition")
st.write("Upload a car photo and this app will detect the license plate and read its characters.")

uploaded_file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    pil_image = Image.open(uploaded_file).convert("RGB")
    car_image = np.array(pil_image)




    st.subheader("Uploaded Image")
    st.image(car_image, width="stretch")

    if not os.path.exists("lpr_model.pkl"):
        st.error("Model file 'lpr_model.pkl' not found in this folder. Train the model first.")
    else:
        with st.spinner("Detecting license plate..."):
            plate = detect_plate(car_image)

        if plate is None:
            st.error("Could not detect a license plate in this image. Try a clearer or closer photo of the vehicle.")
        else:
            st.subheader("Detected Plate")
            st.image(plate, width="content")

            with st.spinner("Segmenting characters..."):
                characters = segment_characters(plate)

            if len(characters) == 0:
                st.error("Plate was found, but no characters could be segmented from it.")
            else:
                st.write("Found " + str(len(characters)) + " character(s)")

                with st.spinner("Recognizing characters..."):
                    model = joblib.load("lpr_model.pkl")
                    plate_number = predict_characters(characters, model)

                st.subheader("Predicted Plate Number")
                st.markdown("## `" + plate_number + "`")
else:
    st.info("Upload an image to get started.")

