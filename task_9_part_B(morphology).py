import cv2
import numpy as np

# Validate

def validate_mask(mask):

    print("validate_mask() executed")

    white = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    coverage = (white / total) * 100

    print("Coverage:", round(coverage, 1), "%")

    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    print("Separate Regions:", len(contours))

    if len(contours) == 1:
        print("PASS")
    else:
        print("FAIL")


# Camera

cap = cv2.VideoCapture(0)

# HSV Range

lower = np.array([0, 50, 50])
upper = np.array([60, 255, 255])

# Kernel

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask

    mask = cv2.inRange(hsv, lower, upper)

    # Opening

    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    # Closing

    clean_mask = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)

    # Result

    result = cv2.bitwise_and(frame, frame, mask=clean_mask)

    # Validate

    validate_mask(clean_mask)

    # Show

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Clean Mask", clean_mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()