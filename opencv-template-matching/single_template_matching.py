# -----------------------------
#   USAGE
# -----------------------------
# python single_template_matching.py --image images/coke_bottle.png --template images/coke_logo.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to the input image file")
ap.add_argument("-t", "--template", type=str, required=True, help="Path to the template image file")
args = vars(ap.parse_args())

# Load the input image and the template image from disk, then display them to the screen
print("[INFO] Loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
cv2.imshow("Image", image)
cv2.imshow("Template", template)

# Convert both the input image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform the template matching procedure
print("[INFO] Performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

# Determine the starting and the ending (x, y) coordinates of the bounding box
(startX, startY) = maxLoc
endX = startX + template.shape[1]
endY = startY + template.shape[0]

# Draw the bounding box on the image
cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
