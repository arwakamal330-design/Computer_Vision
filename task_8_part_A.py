import cv2
import matplotlib.pyplot as plt


img = cv2.imread("img_1.jpg", 0)

if img is None:
    raise FileNotFoundError("Image not found")

# Threshold

thresholds = [
    ("Original", img),
    ("Binary", cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]),
    ("Binary Inv", cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]),
    ("Trunc", cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)[1]),
    ("ToZero", cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)[1]),
    ("ToZero Inv", cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)[1])
]

plt.figure(figsize=(8,6))

for i, (title, image) in enumerate(thresholds):

    plt.subplot(2,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig("Threshold_Grid.png")
plt.show()

# Otsu

otsu_value, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(6,4))

plt.hist(img.ravel(), bins=256)
plt.axvline(otsu_value, color="red")
plt.title("Otsu Histogram")

plt.savefig("Otsu_Histogram.png")
plt.show()

# Adaptive

mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

gaussian = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

adaptive = [("Original", img),("Mean", mean),("Gaussian", gaussian)]

#plt.figure(figsize=(9,3))

for i, (title, image) in enumerate(adaptive):

    plt.subplot(1,3,i+1)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.savefig("Adaptive_Comparison.png")
plt.show()