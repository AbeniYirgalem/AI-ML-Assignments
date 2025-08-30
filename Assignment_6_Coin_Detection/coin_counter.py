import numpy as np
import imutils
import cv2

# Load the image
image = cv2.imread("Assignment_6/coins.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # <Q9-1>

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (11, 11), 0)   # <Q9-2>

# Show original and blurred images
cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Edge detection using Canny
edged = cv2.Canny(blurred, 30, 150)             # <Q9-3>
cv2.imshow("Edges", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # <Q9-4>
cnts = imutils.grab_contours(cnts)

print("I count {} coins in this image".format(len(cnts)))

# Draw contours on the original image
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)  # <Q9-5>
cv2.imshow("Coins Detected", coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract individual coins
for (i, c) in enumerate(cnts):
    # Bounding rectangle
    (x, y, w, h) = cv2.boundingRect(c)
    coin = image[y:y + h, x:x + w]
    print("Coin #{}".format(i + 1))
    cv2.imshow(f"Coin #{i+1}", coin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mask the coin using minimum enclosing circle
    mask = np.zeros(image.shape[:2], dtype="uint8")
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    masked_coin = cv2.bitwise_and(coin, coin, mask=mask)
    cv2.imshow(f"Masked Coin #{i+1}", masked_coin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
