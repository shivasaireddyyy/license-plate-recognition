from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load and preprocess the image
car_image = imread("images/car1.jpg")
car_image_float = img_as_float(car_image)
gray_car_image = rgb2gray(car_image_float)
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

# Label connected regions in the binary image
label_image = measure.label(binary_car_image)

# Tuned to match your actual plate (Region 20: height=65, width=279)
min_height, max_height = 40, 90
min_width, max_width = 200, 350

plate_objects_cordinates = []
fig, ax1 = plt.subplots(1, figsize=(10, 8))
ax1.imshow(gray_car_image, cmap="gray")

for region in regionprops(label_image):
    if region.area < 50:
        continue

    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col

    if (min_height <= region_height <= max_height and
            min_width <= region_width <= max_width and
            region_width > region_height):
        plate_objects_cordinates.append((min_row, min_col, max_row, max_col))
        rect_border = patches.Rectangle(
            (min_col, min_row), region_width, region_height,
            edgecolor="red", linewidth=2, fill=False
        )
        ax1.add_patch(rect_border)
        print(f"Plate candidate: height={region_height}, width={region_width}, position=(row={min_row}, col={min_col})")

print(f"\nFound {len(plate_objects_cordinates)} candidate region(s)")
plt.savefig("plate_detected.png", dpi=150)
plt.show()
