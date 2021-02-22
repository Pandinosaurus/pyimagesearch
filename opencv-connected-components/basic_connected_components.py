# -----------------------------
#   USAGE
# -----------------------------
# python basic_connected_components.py --image license_plate.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
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

# Loop over the number of unique connected component labels
for i in range(0, numLabels):
    # If this is the first component then examine the *background*
    # (typically this component is ignored in the loop)
    if i == 0:
        text = "Examining component {}/{} (background)".format(i+1, numLabels)
    # Otherwise, examine the actual connected component
    else:
        text = "Examining component {}/{}".format(i+1, numLabels)
    # Print a status message update for the current connected component
    print("[INFO] {}".format(text))
    # Extract the connected component statistics and centroid for the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    # Clone the original image (in order to draw on it) and then draw a bounding box surrounding the connected component
    # along with a circle corresponding to the centroid
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    # Construct a mask for the current connected component by finding pixels in the labels array that have the current
    # connected component ID
    componentMask = (labels == i).astype("uint8") * 255
    # Show the output image and the connected component mask
    cv2.imshow("Output", output)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)
