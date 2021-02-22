# -----------------------------
#   USAGE
# -----------------------------
# python filtering_connected_components.py --image license_plate.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4, help="Connectivity for connected component analysis")
args = vars(ap.parse_args())

# Load the input image from disk, convert it to grayscale and threshold it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Apply a connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# Initialize an output mask to store all characters parsed from the license plate
mask = np.zeros(gray.shape, dtype="uint8")

# Loop over the number of unique connected component labels, skipping over the first label (as label zero is background)
for i in range(1, numLabels):
    # Extract the connected component statistics for the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    # Ensure the width, height and area are all neither too small and not too big
    keepWidth = 5 < w < 50
    keepHeight = 45 < h < 65
    keepArea = 500 < area < 1500
    # Ensure the connected component by examining the component passes all three tests
    if all((keepWidth, keepHeight, keepArea)):
        # Construct a mask for the current connected component and then take the bitwise OR with the mask
        print("[INFO] Keeping the connected component {}".format(i))
        componentMask = (labels == i).astype("uint8") * 255
        mask = cv2.bitwise_or(mask, componentMask)

# Show the original input image and the mask for the license plate characters
cv2.imshow("Image", image)
cv2.imshow("Characters", mask)
cv2.waitKey(0)
