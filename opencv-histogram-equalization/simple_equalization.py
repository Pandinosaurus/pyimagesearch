# ------------------------
#   USAGE
# ------------------------
# python simple_equalization.py --image images/moon.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to the input image")
args = vars(ap.parse_args())

# Load the input image from disk and convert it to grayscale
print("[INFO] Loading input image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
print("[INFO] Performing histogram equalization...")
equalized = cv2.equalizeHist(gray)

# Show the original grayscale image and the equalized image
cv2.imshow("Input", gray)
cv2.imshow("Histogram Equalization", equalized)
cv2.waitKey(0)
