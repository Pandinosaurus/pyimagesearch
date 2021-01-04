# -----------------------------
#   USAGE
# -----------------------------
# python opencv_ar_image.py --image examples/input_01.jpg --source sources/squirrel.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image containing ArUCo tag")
ap.add_argument("-s", "--source", required=True, help="Path to input source image that will be put on input")
args = vars(ap.parse_args())

# Load the input image from disk, resize it and grab its spatial dimensions
print("[INFO] Loading input image and source image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(imgH, imgW) = image.shape[:2]

# Load the source image from disk
source = cv2.imread(args["source"])

# Load the ArUCo dictionary, grab the ArUCo parameters and detect the markers in the image
print("[INFO] Detecting markers...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# If four markers have not been found in the input image then it is not possible to apply augmented reality
if len(corners) != 4:
    print("[INFO] Could not find 4 corners in the input image! Exiting...")
    sys.exit(0)

# Otherwise, once the four ArUCo markers have been found it is possible to continue by flatenning the ArUCo IDs list
# and initialize the list of reference points
print("[INFO] Constructing augmented reality visualization...")
ids = ids.flatten()
refPts = []

# Loop over the IDs of the ArUCo markers in the following order: top-left -> top-right -> bottom-right -> bottom-left
for i in (923, 1001, 241, 1007):
    # Grab the index of the corner with the current ID and append the corner (x, y)-coordinates
    # to the list of reference points
    j = np.squeeze(np.where(ids == i))
    corner = np.squeeze(corners[j])
    refPts.append(corner)

# Unpack the ArUCo reference points and use them to define the *destination* transform matrix, making sure that the
# points are specified in the following order: top-left -> top-right -> bottom-right -> bottom-left
(refPtTL, refPtTR, refPtBR, refPtBL) = refPts
dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
dstMat = np.array(dstMat)

# Grab the spatial dimensions of the source image and define the transform matrix for the *source* image in the
# following order: top-left -> top-right -> bottom-right -> bottom-left
(srcH, srcW) = source.shape[:2]
srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

# Compute the homography matrix and then wrap the source image to its destination based on the homography
(H, _) = cv2.findHomography(srcMat, dstMat)
warped = cv2.warpPerspective(source, H, (imgW, imgH))

# Construct a mask for the source image now that the perspective warp has been taken place
# (This mask is needed to copy the source image into the destination)
mask = np.zeros((imgH, imgW), dtype="uint8")
cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

# This step is optional, but to give the source image a black border surrounding it when applied to the source image,
# it is possible to apply a dilation operation
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, rect, iterations=2)

# Create a three channel version of the mask by stacking it depth-wise, such that it is possible to copy the warped
# source image into the input image
maskScaled = mask.copy() / 255.0
maskScaled = np.dstack([maskScaled] * 3)

# Copy the warped source image unto the input image by:
# (1) Multiplying the warped image and the masked image together;
# (2) Multiplying the original input image with the mask;
# (giving more weight to the input where there *ARE NOT* masked pixels)
# (3) Adding the resulting multiplication results together
warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
output = cv2.add(warpedMultiplied, imageMultiplied)
output = output.astype("uint8")

# Show the input image, source image and the output of the augmented reality
cv2.imshow("Input", image)
cv2.imshow("Source", source)
cv2.imshow("OpenCV AR Output", output)
cv2.waitKey(0)
