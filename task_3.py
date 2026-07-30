import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_1.jpg")

if img is None:
    raise FileNotFoundError

# convert image to all 5 spaces ==================================
bgr = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

# display all 5 in one matplotlib ================================
images = [bgr, gray, hsv, lab, ycrcb]
titles = ['original images','gray','hsv','lab','ycrcb']
cmaps = [None,'gray',None,None,None]

fig,axis = plt.subplots(1,5,figsize=(12,6))

for ax,image,title,cmap in zip (axis,images,titles,cmaps):
    ax.imshow (image,cmap=cmap)
    ax.set_title (title)
    ax.axis ('off')
plt.tight_layout()
plt.show()

# center pixel value =============================================
for image,title in zip (images,titles):
    print (title,":",f'{image[image.shape[0]//2,image.shape[1]//2]}')

# isolate the object ==============================================
Lower_green = np.array([10,50,50])
Upper_green = np.array([40,255,255])
mask = cv2.inRange(hsv,Lower_green,Upper_green)
bitwise = cv2.bitwise_and(img,img,mask=mask)
result = cv2.cvtColor(bitwise,cv2.COLOR_BGR2RGB)

imgs = [bgr,mask,result]
tits = ['original images','mask','result']
cmaps = [None,'gray',None]

fig,axis = plt.subplots(1,3,figsize=(10,6))

for ax,image,title,cmap in zip (axis,imgs,tits,cmaps):
    ax.imshow (image,cmap=cmap)
    ax.set_title (title)
    ax.axis ('off')
plt.tight_layout()
plt.show()

# a] Hue range used: 10:40

# b] these S and V bounds: 
#    S bounds were used to remove low-saturation (faded) colours
#    V bounds were used to remove very dark or very bright pixels, improving segmentation accuracy
# c] No, I didn't need two inRange call
