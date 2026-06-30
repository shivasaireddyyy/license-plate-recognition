from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Load the cropped plate
plate_image = imread("cropped_plate.png")
plate_float = img_as_float(plate_image)
gray_plate = rgb2gray(plate_float)

# Threshold to binary — characters are usually darker than the plate background
threshold_value = threshold_otsu(gray_plate)
binary_plate = gray_plate < threshold_value  # note: < because characters are dark

label_image = measure.label(binary_plate)

plate_height, plate_width = gray_plate.shape

# Character size heuristics relative to plate size — we may need to tune these
char_min_height = 0.35 * plate_height
char_max_height = 0.95 * plate_height
char_min_width = 0.03 * plate_width
char_max_width = 0.3 * plate_width

characters = []
column_positions = []

fig, ax1 = plt.subplots(1, figsize=(10, 4))
ax1.imshow(gray_plate, cmap="gray")

for region in regionprops(label_image):
    min_row, min_col, max_row, max_col = region.bbox
    h = max_row - min_row
    w = max_col - min_col

    if char_min_height <= h <= char_max_height and char_min_width <= w <= char_max_width:
        char_image = binary_plate[min_row:max_row, min_col:max_col]
        characters.append(char_image)
        column_positions.append(min_col)

        rect_border = patches.Rectangle(
            (min_col, min_row), w, h,
            edgecolor="red", linewidth=1, fill=False
        )
        ax1.add_patch(rect_border)
        print(f"Character found: height={h}, width={w}, col={min_col}")

print(f"\nFound {len(characters)} character(s)")
plt.savefig("characters_detected.png", dpi=150)
plt.show()

# Save each character, sorted left-to-right (by column position)
if characters:
    os.makedirs("segmented_characters", exist_ok=True)
    sorted_chars = [c for _, c in sorted(zip(column_positions, characters), key=lambda pair: pair[0])]
    for i, char in enumerate(sorted_chars):
        # Resize to a standard size for later ML use
        resized = resize(char, (20, 20))
        imsave(f"segmented_characters/char_{i}.png", (resized * 255).astype(np.uint8))
    print(f"Saved {len(sorted_chars)} character images to segmented_characters/")
