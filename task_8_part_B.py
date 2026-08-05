import cv2
import numpy as np

# validate

def validate_mask(mask):

    print("validate_mask() executed")

    white = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    coverage = (white / total) * 100

    print("Coverage:", round(coverage, 1), "%")

    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print("Separate regions:", len(contours))

    if len(contours) == 1:
        print("PASS")
    else:
        print("FAIL")

    print("------------------")


# Camera

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

    # Validate

    validate_mask(mask)

    # Show

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    # Exit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
