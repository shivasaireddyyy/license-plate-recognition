from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_float
import matplotlib.pyplot as plt

car_image = imread("images/car1.jpg")
print("Image shape:", car_image.shape)
print("Image dtype:", car_image.dtype)

# Convert to float (values scaled to 0-1) before grayscale conversion
car_image_float = img_as_float(car_image)
gray_car_image = rgb2gray(car_image_float)

threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(gray_car_image, cmap="gray")
ax1.set_title("Grayscale Image")
ax2.imshow(binary_car_image, cmap="gray")
ax2.set_title("Binary Image")
plt.show()
