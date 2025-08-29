import numpy as np
import cv2

# Load the image
image = cv2.imread("Assignment_5_Coin_Detection/coins.png")
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale and blur
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
print("Blurred")
cv2.imshow("Blurred", image_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Canny edge detection
canny = cv2.Canny(image_blur, 30, 200)  # 30, 150
print("Canny")
cv2.imshow("Canny Edges", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
