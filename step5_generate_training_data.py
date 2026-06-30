from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

output_dir = "train_data"
os.makedirs(output_dir, exist_ok=True)

font_paths = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

available_fonts = [f for f in font_paths if os.path.exists(f)]
print("Found " + str(len(available_fonts)) + " usable font(s): " + str(available_fonts))

if not available_fonts:
    print("No fonts found at expected paths, using PIL default font instead")
    available_fonts = [None]

count = 0
for char in characters:
    char_dir = os.path.join(output_dir, char)
    os.makedirs(char_dir, exist_ok=True)

    for i, font_path in enumerate(available_fonts):
        img = Image.new("L", (40, 40), color=255)
        draw = ImageDraw.Draw(img)

        if font_path:
            font = ImageFont.truetype(font_path, 30)
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        position = ((40 - text_w) / 2 - bbox[0], (40 - text_h) / 2 - bbox[1])

        draw.text(position, char, fill=0, font=font)

        arr = np.array(img)
        resized = resize(arr, (20, 20))
        binary = (resized < 0.5).astype(np.uint8) * 255

        imsave(os.path.join(char_dir, char + "_" + str(i) + ".png"), binary)
        count += 1

print("Generated " + str(count) + " training images across " + str(len(characters)) + " characters")
print("Saved to " + output_dir + "/")
