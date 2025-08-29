import numpy as np
import cv2

image = cv2.imread("Assignment_4_Image_Blurring/trex.png")
print("Original")
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Averaging blur
blurred = np.hstack([
    cv2.blur(image, (3, 3)),
    cv2.blur(image, (5, 5)),
    cv2.blur(image, (7, 7))
])
print("Averaged")
cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian blur
blurred = np.hstack([
    cv2.GaussianBlur(image, (3, 3), 0),
    cv2.GaussianBlur(image, (5, 5), 0),
    cv2.GaussianBlur(image, (7, 7), 0)
])
print("Gaussian")
cv2.imshow("Gaussian", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Median blur
blurred = np.hstack([
    cv2.medianBlur(image, 3),
    cv2.medianBlur(image, 5),
    cv2.medianBlur(image, 7)
])
print("Median")
cv2.imshow("Median", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Bilateral filter
blurred = np.hstack([
    cv2.bilateralFilter(image, 5, 21, 21),
    cv2.bilateralFilter(image, 7, 31, 31),
    cv2.bilateralFilter(image, 9, 41, 41)
])
print("Bilateral")
cv2.imshow("Bilateral", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()


