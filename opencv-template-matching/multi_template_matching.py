# -----------------------------
#   USAGE
# -----------------------------
# python multi_template_matching.py --image images/8_diamonds.png --template images/diamonds_template.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to the input image file")
ap.add_argument("-t", "--template", type=str, required=True, help="Path to the template image file")
ap.add_argument("-b", "--threshold", type=float, default=0.8, help="Threshold for multi-template matching")
args = vars(ap.parse_args())

# Load the input image and template image from disk, then grab the template image spatial dimensions
print("[INFO] Loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]

# Display the  image and template to the screen
cv2.imshow("Image", image)
cv2.imshow("Template", template)

# Convert both the image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
print("[INFO] Performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)

# Find all locations in the result map where the matched value is greater than the threshold,
# then clone the original image in order to draw on it
(yCoords, xCoords) = np.where(result >= args["threshold"])
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

# Loop over the starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
    # Draw the bounding box on the image
    cv2.rectangle(clone, (x, y), (x + tW, y + tH), (255, 0, 0), 3)

# Show the output image *before* applying non-maxima suppression
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)

# Initialize the list of rectangles
rects = []

# Loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
    # Update the list of rectangles
    rects.append((x, y, x + tW, y + tH))

# Apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

# Loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
    # Draw the bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)
