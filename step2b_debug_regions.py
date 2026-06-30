from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
image_path = sys.argv[1] if len(sys.argv) > 1 else "images/car1.jpg"
car_image = imread(image_path)
car_image_float = img_as_float(car_image)
gray_car_image = rgb2gray(car_image_float)
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

label_image = measure.label(binary_car_image)

print(f"Image size (height x width): {label_image.shape[0]} x {label_image.shape[1]}")
print()

fig, ax1 = plt.subplots(1, figsize=(12, 9))
ax1.imshow(gray_car_image, cmap="gray")

# Show ALL regions above a minimum area, in yellow, with their sizes printed
count = 0
for region in regionprops(label_image):
    if region.area < 100:
        continue
    min_row, min_col, max_row, max_col = region.bbox
    h = max_row - min_row
    w = max_col - min_col

    # only show reasonably plate-shaped candidates (wider than tall, not tiny, not huge)
    if w > h and h > 10 and w < label_image.shape[1] * 0.6:
        count += 1
        print(f"Region {count}: height={h}, width={w}, area={region.area}, position=(row={min_row}, col={min_col})")
        rect_border = patches.Rectangle(
            (min_col, min_row), w, h,
            edgecolor="yellow", linewidth=1, fill=False
        )
        ax1.add_patch(rect_border)

plt.title("All candidate regions (yellow) - note the height/width of the one on your actual plate")
plt.savefig("debug_regions.png", dpi=150)
print()
print("Saved visualization to debug_regions.png")
plt.show()
