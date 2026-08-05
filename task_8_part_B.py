import cv2
import numpy as np

#def validate_mask(mask):



cap = cv2.VideoCapture(0)

lower = np.array([35, 50, 50])
upper = np.array([85, 255, 255])

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask

    mask = cv2.inRange(hsv, lower, upper)

    # Result

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()