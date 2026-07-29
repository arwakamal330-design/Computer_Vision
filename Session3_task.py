import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread ("img_1.jpg")

if img is None :
    raise FileNotFoundError
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split (hsv)
    print ("hsv range",h.min(),"-",h.max())
    print ("hsv center ", hsv[hsv.shape[0]//2,hsv.shape[1]//2])
    print (h)
    print (s)
    print (v)

    lower_green = np.array ([15,50,50])
    upper_green = np.array ([40,255,255])

    mask = cv2.inRange (hsv,lower_green,upper_green)
    mask_inv =cv2.bitwise_not(mask)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.add(L, 50)

    lab_bright = cv2.merge((L, A, B))

    bright_img = cv2.cvtColor(lab_bright, cv2.COLOR_LAB2BGR)



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    foreground = cv2.bitwise_and(bright_img, bright_img, mask=mask)
    
    background = cv2.bitwise_and(gray_bgr, gray_bgr, mask=mask_inv)
    
    final = cv2.add(foreground, background)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bright_rgb = cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    ax[0,0].imshow(img_rgb)
    ax[0,0].set_title("Original")

    ax[0,1].imshow(mask, cmap="gray")
    ax[0,1].set_title("Mask")

    ax[0,2].imshow(bright_rgb)
    ax[0,2].set_title("LAB Bright")

    ax[1,0].imshow(gray_rgb)
    ax[1,0].set_title("Gray Background")

    ax[1,1].imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
    ax[1,1].set_title("Foreground")

    ax[1,2].imshow(final_rgb)
    ax[1,2].set_title("Final Result")

    for a in ax.ravel():
        a.axis("off")
        
    plt.imshow (final)
    plt.tight_layout()
    plt.savefig("Task3_1_Result.jpg")
    plt.show()
