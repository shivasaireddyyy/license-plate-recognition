from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.morphology import dilation, square

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

output_dir = "train_data"
os.makedirs(output_dir, exist_ok=True)

font_paths = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
available_fonts = [f for f in font_paths if os.path.exists(f)]
print("Found " + str(len(available_fonts)) + " usable font(s)")

font_sizes = [24, 28, 32, 36]
offsets = [(-2, -2), (0, 0), (2, 2), (-2, 2)]
thickness_levels = [0, 1, 2]

count = 0
for char in characters:
    char_dir = os.path.join(output_dir, char)
    os.makedirs(char_dir, exist_ok=True)

    variant = 0
    for font_path in available_fonts:
        for size in font_sizes:
            for offset in offsets:
                img = Image.new("L", (40, 40), color=255)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(font_path, size)

                bbox = draw.textbbox((0, 0), char, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                position = (
                    (40 - text_w) / 2 - bbox[0] + offset[0],
                    (40 - text_h) / 2 - bbox[1] + offset[1]
                )
                draw.text(position, char, fill=0, font=font)

                arr = np.array(img)
                mask = arr < 128
                if mask.any():
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    cropped = arr[rmin:rmax+1, cmin:cmax+1]
                else:
                    cropped = arr
                resized = resize(cropped, (20, 20))
                binary = (resized < 0.5).astype(np.uint8)

                for thickness in thickness_levels:
                    final_binary = binary
                    if thickness > 0:
                        final_binary = dilation(binary, square(thickness + 1))

                    output_img = (final_binary * 255).astype(np.uint8)
                    imsave(os.path.join(char_dir, char + "_" + str(variant) + ".png"), output_img)
                    variant += 1
                    count += 1

print("Generated " + str(count) + " training images across " + str(len(characters)) + " characters")
print("That is " + str(count // len(characters)) + " variations per character")