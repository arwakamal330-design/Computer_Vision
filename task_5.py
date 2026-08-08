import cv2
import numpy as np
import matplotlib.pyplot as plt


clip_limit = 2.0
tile_grid_size = (8,8)

img = cv2.imread("img_1.jpg")

# Part A: Diagnosis =========================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist_gray = cv2.calcHist([gray], [0], None, [256], [0,256])

mean_brightness = np.mean(gray)
std_brightness = np.std(gray)

print(f'Mean brightness: {mean_brightness:.2f}')
print(f'Standard deviation: {std_brightness:.2f}')


plt.figure(figsize=(8,5))
plt.plot(hist_gray, color='black')
plt.xlim([0,256])
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.show()

# The histogram is shifted toward the bright side, so the image is relatively bright


colors = ('b', 'g', 'r')

plt.figure(figsize=(8,5))

for i, col in enumerate(colors):

    hist = cv2.calcHist([img], [i], None, [256], [0,256])

    plt.plot(hist, color=col)

plt.xlim([0,256])
plt.title("BGR Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")
plt.legend(["Blue", "Green", "Red"])
plt.show()


# Part B: Enhancement =======================


equalized = cv2.equalizeHist(gray)


clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)

clahe_gray = clahe.apply(gray)


lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

L, A, B = cv2.split(lab)

L_enhanced = clahe.apply(L)

lab_enhanced = cv2.merge([L_enhanced, A, B])

clahe_color = cv2.cvtColor(lab_enhanced,cv2.COLOR_LAB2BGR)


images = [gray, equalized, clahe_gray]

histograms = []

for image in images:

    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    histograms.append(hist)


fig, axis = plt.subplots(2,3, figsize=(12,7))


image_names = ["Original Grayscale","equalizeHist","CLAHE"]

for i in range(3):

    axis[0,i].imshow(images[i],cmap='gray')

    axis[0,i].set_title(image_names[i])
    axis[0,i].axis('off')


hist_names = ["Original Histogram","equalizeHist Histogram","CLAHE Histogram"]

for i in range(3):

    axis[1,i].plot(histograms[i],color='black')

    axis[1,i].set_title(hist_names[i])
    axis[1,i].set_xlim([0,256])

plt.tight_layout()

plt.savefig( "comparison.png",dpi=150,bbox_inches='tight')

plt.show()


# CLAHE gives better results because it shows more details and improves the contrast