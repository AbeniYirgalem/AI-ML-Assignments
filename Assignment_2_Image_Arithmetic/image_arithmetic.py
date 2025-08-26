import numpy as np
import cv2

# Load image
image = cv2.imread("trex.png")
cv2.imshow("Original Image", image)

# -----------------------------
# NumPy operations
# -----------------------------

# 200 + 100
result_add = np.add(np.array([[200]]), np.array([[100]]))
print("NumPy add (200+100): {}".format(result_add))

# 50 - 100
result_subtract = np.subtract(np.array([[50]]), np.array([[100]]))
print("NumPy subtract (50-100): {}".format(result_subtract))

# -----------------------------
# OpenCV operations
# -----------------------------

# 200 + 100 ==> 300 ==> 255
result_add = cv2.add(np.uint8([[200]]), np.uint8([[100]]))
print("OpenCV add (200+100): {}".format(result_add))

# 50 - 100 ==> -50 ==> 0
result_subtract = cv2.subtract(np.uint8([[50]]), np.uint8([[100]]))
print("OpenCV subtract (50-100): {}".format(result_subtract))

# -----------------------------
# Wrap around demonstration
# -----------------------------

x = np.uint8([250])
y = np.uint8([10])

print("cv2.add(x,y) = {}".format(cv2.add(x,y))) # 250+10=260 -> 255
print("NumPy x+y = {}".format(x+y))             # 260 % 256 = 4

print("Wrap around (200+100): {}".format(np.uint8([200]) + np.uint8([100])))
print("Wrap around (50-100): {}".format(np.uint8([50]) - np.uint8([100])))

# -----------------------------
# Brightness adjustment with OpenCV
# -----------------------------

M = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added (Brightness Increased)", added)

# Subtract (darken)
M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted (Brightness Decreased)", subtracted)

# -----------------------------
# Matrix subtraction example
# -----------------------------
print("Matrix subtraction: \n", cv2.subtract(np.ones([2,3], dtype="uint8") * 100, 
                                             np.ones([2,3], dtype="uint8") * 30))

print("Image shape: ", image.shape)

# -----------------------------
# Wait for a key press and close windows
# -----------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
