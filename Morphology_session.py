import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img_3.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Kernels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

# Morphology
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_2)

gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, kernel)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_2)

blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_2)

# Show
fig, axis = plt.subplots(2,4, figsize=(12,6))

images = [gray, mask, opening, closing, gradient, tophat, blackhat]

names = ["Gray", "Mask", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]

for ax, image, name in zip(axis.ravel(), images, names):
    ax.imshow(image, cmap="gray")
    ax.set_title(name)
plt.axis("off")

axis[1,3].axis("off")

plt.tight_layout()
plt.show()