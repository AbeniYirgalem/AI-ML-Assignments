import matplotlib.pyplot as plt
import cv2

# Read image
image = cv2.imread("Assignment1_ImageProcessing/trex.png")
if image is None:
	print("Error: Could not load 'trex.png'. Check the file path and format.")
	exit(1)

# Print details
print(image.shape)
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {} pixels".format(image.shape[2]))

# Show image using OpenCV
cv2.imshow("T-Rex Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()

# Load image
image2 = cv2.imread("Assignment1_ImageProcessing/mountain_nyala.jpg")
if image2 is None:
	print("Error: Could not load 'mountain_nyala.jpg'. Check the file path and format.")
	exit(1)

# Show with OpenCV
cv2.imshow("Mountain Nyala", image2)
cv2.waitKey(0)

# Show with matplotlib (converted to RGB)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Access a pixel
(b, g, r) = image2[90, 219]   # y=90, x=219
print("Pixel at (219, 90) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# Draw a white rectangle on ROI
head = (300, 300)
tail = (500, 500)
image2[head[1]:tail[1], head[0]:tail[0]] = (255, 255, 255)

# Show the modified image
cv2.imshow("Edited Image", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()