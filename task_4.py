import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img_1.jpg")

H, W, C = img.shape
print (f'[original image: Height, Weight, Channels]:{img.shape}') # (367, 550, 3)

# 1] Crop :--------------------

roi = img[100:270,100:450].copy()

print (f'[roi]:{img.shape}')
# 2] Resize :------------------

resize_1 = cv2.resize(img,(W//2,H//2),interpolation=cv2.INTER_AREA)

resize_2 = cv2.resize(img,(2*W,2*H),interpolation=cv2.INTER_CUBIC)

resize_3 = cv2.resize(img,(200,200),interpolation=cv2.INTER_LINEAR)

print (f'[resize_1: Height, Weight, Channels]:{resize_1.shape}')
print (f'[resize_2: Height, Weight, Channels]:{resize_2.shape}')
print (f'[resize_3: Height, Weight, Channels]:{resize_3.shape}')

# 3] Rotate :-----------------

M_1 = cv2.getRotationMatrix2D (center=(W//2,H//2),angle=30,scale=1)
rotate_1 = cv2.warpAffine(img,M_1,(W,H))

M_2 = cv2.getRotationMatrix2D (center=(W//2,H//2),angle=-30,scale=1)
rotate_2 = cv2.warpAffine(img,M_2,(W,H))

# 4] Flip :------------------

flip_1 = cv2.flip (img,0)
flip_2 = cv2.flip (img,1)
flip_3 = cv2.flip (img,-1)

# 5] Blend :----------------

Overlay = img.copy()
img_2 = cv2.imread("img_9.jpg")
resize_img_2 = cv2.resize(img_2,(W,H))
blend_1 = cv2.addWeighted(Overlay,0.25,resize_img_2,0.75,0)
blend_2 = cv2.addWeighted(Overlay,0.5,resize_img_2,0.5,0)
blend_3 = cv2.addWeighted(Overlay,0.75,resize_img_2,0.25,0)

# 6] binary mask :---------

img_3 = cv2.imread("img_8.jpg")
resize_img_3 = cv2.resize(img_3,(W,H))

mask = np.zeros((H,W), dtype=np.uint8)

cv2.rectangle(mask, (100,100), (300,300), 255, -1)

cv2.circle(mask, (450,250), 100, 255, -1)

mask_inv = cv2.bitwise_not(mask)

foreground = cv2.bitwise_and(resize_img_3, resize_img_3, mask=mask)

background = cv2.bitwise_and(img, img, mask=mask_inv)

final = cv2.add(background, foreground)


# Display Photos and Save ========================================================

fig , axis = plt.subplots (3,5,figsize=(12,6))
images = [img,roi,resize_1,resize_2,resize_3,rotate_1,rotate_2,flip_1,flip_2,flip_3,blend_1,
          blend_2,blend_3,resize_img_3,final]
names = ["original image","ROI","Resize_1","Resize_2","Resize_3","rotate_30","rotate_-30",
         "flip around X","flip around Y","flip around radius","Blend_0.25","Blend_0.5",
         "Blend_0.75","mask_img","img with mask"]
for ax, im, name in zip (axis.ravel(),images,names):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    ax.set_title (name)
    ax.axis('off')
plt.tight_layout()
plt.savefig("Output.png",dpi=150,bbox_inches="tight")
plt.show()

