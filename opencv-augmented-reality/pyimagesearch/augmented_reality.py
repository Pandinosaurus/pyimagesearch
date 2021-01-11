# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import cv2

# -----------------------------
#   VARIABLES
# -----------------------------
# Initialize the cached reference points
CACHED_REF_PTS = None


# -----------------------------
#   FUNCTIONS
# -----------------------------
def find_and_warp(frame, source, cornerIDs, arucoDict, arucoParams, useCache=False):
    # Grab a reference to the cached reference points
    global CACHED_REF_PTS
    # Grab the width and height of the frame and source image respectively
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]
    # Detect ArUCo markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    # If the four ArUCo markers have not been found yet, initialize an empty IDs list, otherwise flatten the IDs list
    ids = np.array([]) if len(corners) != 4 else ids.flatten()
    # Initialize the list of reference points
    refPts = []
    # Loop over the IDs of the ArUCo markers in following order: top-left -> top-right -> bottom-right -> bottom-left
    for i in cornerIDs:
        # Grab the index of the corner with the current ID
        j = np.squeeze(np.where(ids == i))
        # If an empty list is received instead of an integer index list, then find the marker with the current ID
        if j.size == 0:
            continue
        # Otherwise, append the corner (x, y) coordinates to the list of reference points
        corner = np.squeeze(corners[j])
        refPts.append(corner)
    # Check to see if the four ArUCo markers have not been found yet
    if len(refPts) != 4:
        # If the cached reference points are allowed to be used, fall back on them
        if useCache and CACHED_REF_PTS is not None:
            refPts = CACHED_REF_PTS
        # Otherwise, it is not possible to use the cache and/or there are no previous cached reference points,
        # and so return early
        else:
            return None
    # If the cached reference points are allowed to be used, then update the cache with the current set
    if useCache:
        CACHED_REF_PTS = refPts
    # Unpack the ArUCo reference points and use those reference points to define the *destination* transform matrix,
    # making sure the points are specified in the following order: top-left -> top-right -> bottom-right -> bottom-left
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    # Define the transform matrix for the *source* image in the following order:
    # top-left -> top-right -> bottom-right -> bottom-left
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    # Compute the homography matrix and then warp the source image to the destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    # Construct a mask for the source image now that the perspective warp has taken place
    # ( This mask will be used to copy the source image into its destination )
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)
    # This step is optional, but to give the source image a black border surrounding it, you can simply apply a dilation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    # Create a three channel version of the mask by stacking it depth-wise, such that it is possible to copy the warped
    # source image into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    # Copy the warped source image into the input image by:
    # (1) Multiplying the warped image and the masked image together;
    # (2) Multiplying the original input image with the mask (giving more weight to the input where there are no masked
    # pixels);
    # (3) Adding the resulting multiplications together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(frame.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    # Return the output frame to the calling function
    return output
