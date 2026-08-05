import cv2
import matplotlib.pyplot as plt

# Part A

img = cv2.imread("img_8.jpg", 0)

if img is None:
    raise FileNotFoundError("Image not found")

# Kernels

rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Morphology

erode = cv2.erode(img, rect)

dilate = cv2.dilate(img, rect)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, rect)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, rect)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, ellipse)

blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ellipse)

# Display

titles = ["gray","Erode","Dilate","Open","Close","Gradient","Top Hat","Black Hat"]

images = [img,erode,dilate,opening,closing,gradient,tophat,blackhat]

fig, axes = plt.subplots(2, 4, figsize=(10, 6))

for ax, image, title in zip(axes.ravel(), images, titles):

    ax.imshow(image, cmap="gray")

    ax.set_title(title)

    ax.axis("off")

plt.tight_layout()

plt.savefig("Morphology_Grid.png")

plt.show()