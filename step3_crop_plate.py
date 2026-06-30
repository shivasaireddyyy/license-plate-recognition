from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt

car_image = imread("images/car1.jpg")
car_image_float = img_as_float(car_image)
gray_car_image = rgb2gray(car_image_float)
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

label_image = measure.label(binary_car_image)

min_height, max_height = 40, 90
min_width, max_width = 200, 350

plate_like_objects = []

for region in regionprops(label_image):
    if region.area < 50:
        continue
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col

    if (min_height <= region_height <= max_height and
            min_width <= region_width <= max_width and
            region_width > region_height):
        # Crop directly from the original color image, with a small margin
        margin = 2
        plate_like_objects.append(
            car_image[max(0, min_row-margin):max_row+margin,
                       max(0, min_col-margin):max_col+margin]
        )

print(f"Found {len(plate_like_objects)} plate candidate(s)")

if plate_like_objects:
    plate = plate_like_objects[0]
    imsave("cropped_plate.png", plate)
    print("Saved cropped plate to cropped_plate.png")

    plt.imshow(plate)
    plt.title("Cropped License Plate")
    plt.show()
else:
    print("No plate found — check the size thresholds")
