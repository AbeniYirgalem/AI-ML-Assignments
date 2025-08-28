import numpy as np
import cv2

# Load the image
image = cv2.imread("Assignment_3_ImageMasking/beach.png")
cv2.imshow("Original image", image)

# Create a black mask and draw a white rectangle in the center
mask = np.zeros(image.shape[:2], dtype="uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75 , cY + 75), 255, -1)
cv2.imshow("Rectangle mask", mask)

# Apply the rectangle mask to the image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Rectangle masked image", masked)

# Create a new mask with a circle
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (cX, cY), 100, 255, -1)
cv2.imshow("Circle mask", mask)

# Apply the circle mask to the image
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Circle masked image", masked)

# Wait until a key is pressed and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
