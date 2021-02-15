# -----------------------------
#   USAGE
# -----------------------------
# python color_correction.py --reference reference.jpg --input examples/01.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import sys


# -----------------------------
#   FUNCTIONS
# -----------------------------
def find_color_card(image):
    # Load the ArUCo dictionary, grab the ArUCo parameters and detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # Try to extract the coordinates of the color correction card
    try:
        # Otherwise, this means that the four ArUCo markers have been found and
        # so continue by flattening the ArUCo IDs list
        ids = ids.flatten()
        # Extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]
        # Extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]
        # Extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]
        # Extract the bottom left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]
    # The color correction card could not be found, so gracefully return
    except:
        return None
    # Build the list of reference points and apply a perspective transform to obtain a top-down,
    # birds-eye-view of the color matching card
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    # Return the color matching card to the calling function
    return card


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True, help="Path to the input reference image")
ap.add_argument("-i", "--input", required=True, help="Path to the input image to apply color correction to")
args = vars(ap.parse_args())

# Load the reference image and input images from disk
print("[INFO] Loading images...")
ref = cv2.imread(args["reference"])
image = cv2.imread(args["input"])

# Resize the reference and input images
ref = imutils.resize(ref, width=600)
image = imutils.resize(image, width=600)

# Display the reference and input images to the screen
cv2.imshow("Reference", ref)
cv2.imshow("Input", image)

# Find the color matching card in each image
print("[INFO] Finding color matching cards...")
refCard = find_color_card(ref)
imageCard = find_color_card(image)

# If the color matching card is not found in either the reference or the input image, gracefully exit the program
if refCard is None or imageCard is None:
    print("[INFO] Could not find color matching cards in both images! Exiting...")
    sys.exit(0)

# Show the color matching card in the reference image and the in the input image respectively
cv2.imshow("Reference Color Card", refCard)
cv2.imshow("Input Color Card", imageCard)

# Apply histogram matching from the color matching card in the reference image
# to the color matching card in the input image
print("[INFO] Matching images...")
imageCard = exposure.match_histograms(imageCard, refCard, multichannel=True)

# Show the input color matching card after histogram matching
cv2.imshow("Input Color Card After Matching", imageCard)
cv2.waitKey(0)

