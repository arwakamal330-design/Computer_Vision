import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)

    abs_x = cv2.convertScaleAbs(sobelx)
    abs_y = cv2.convertScaleAbs(sobely)
    abs_mag = cv2.convertScaleAbs(magnitude)

    # Laplacian
    laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplace_blur = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)

    laplace = cv2.convertScaleAbs(laplace)
    laplace_blur = cv2.convertScaleAbs(laplace_blur)

    # Canny
    edges = cv2.Canny(blur, 50, 150)

    # Overlay
    overlay = frame.copy()
    overlay[edges == 255] = (0, 255, 0)

    cv2.imshow("Gray", gray)
    cv2.imshow("Sobel X", abs_x)
    cv2.imshow("Sobel Y", abs_y)
    cv2.imshow("Magnitude", abs_mag)
    cv2.imshow("Laplacian", laplace)
    cv2.imshow("LoG", laplace_blur)
    cv2.imshow("Canny", edges)
    cv2.imshow("Overlay", overlay)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        cv2.imwrite("Frame.png", frame)
        cv2.imwrite("Edges.png", edges)
        cv2.imwrite("Overlay.png", overlay)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()