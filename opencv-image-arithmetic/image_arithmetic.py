# -----------------------------
#   USAGE
# -----------------------------
# python image_arithmetic.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="grand_canyon.png", help="Path to the input image")
args = vars(ap.parse_args())

# Images are NumPy arrays stored as unsigned 8-bit integers (uint8) with values in the range [0, 255];
# When using the add/subtract functions in OpenCV, these values will be *clipped* to this range, even if they fall
# outside the range [0, 255] after applying the operation
added = cv2.add(np.uint8([200]), np.uint8([100]))
subtracted = cv2.subtract(np.uint8([50]), np.uint8([100]))
print("[INFO] Max of 255: {}".format(added))
print("[INFO] Min of 0: {}".format(subtracted))

# Using NumPy arithmetic operations (rather than OpenCV operations) will result in a modulo ("wrap around") instead of
# being clipped to the range [0, 255]
added = np.uint8([200]) + np.uint8([100])
subtracted = np.uint8([50]) - np.uint8([100])
print("[INFO] Wrap around: {}".format(added))
print("[INFO] Wrap around: {}".format(subtracted))

# Load the original input image and display it to the screen
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Increasing the pixel intensities in the input image by 100 is accomplished by constructing a NumPy array that has the
# *same dimensions* as the input image, filling it with ones, multiplying it by 100, and then adding the input image
# and matrix together
M = np.ones(image.shape, dtype="uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Lighter", added)

# Similarly, subtract 50 from all pixels in the image and make it darker
M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Darker", subtracted)
cv2.waitKey(0)