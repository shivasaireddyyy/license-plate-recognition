from skimage.io import imread
import matplotlib.pyplot as plt

train_img = imread("train_data/A/A_0.png", as_gray=True)
real_img = imread("segmented_characters/char_0.png", as_gray=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(train_img, cmap="gray")
ax1.set_title("Training image (letter A)")
ax2.imshow(real_img, cmap="gray")
ax2.set_title("Real segmented character")
plt.savefig("comparison.png", dpi=150)
plt.show()

print("Training image pixel range: min=" + str(train_img.min()) + " max=" + str(train_img.max()))
print("Real image pixel range: min=" + str(real_img.min()) + " max=" + str(real_img.max()))
print("Training image mean brightness: " + str(train_img.mean()))
print("Real image mean brightness: " + str(real_img.mean()))
