# ------------------------
#   USAGE
# ------------------------
#  python guess_aruco_type.py --image images/example_01.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image containing ArUCo tag")
args = vars(ap.parse_args())

# Define the names of each possible ArUCo tag that the OpenCV supports
ARUCO_DICT = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50, "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
              "DICT_4X4_250": cv2.aruco.DICT_4X4_250, "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
              "DICT_5X5_50": cv2.aruco.DICT_5X5_50, "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
              "DICT_5X5_250": cv2.aruco.DICT_5X5_250, "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
              "DICT_6X6_50": cv2.aruco.DICT_6X6_50, "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
              "DICT_6X6_250": cv2.aruco.DICT_6X6_250, "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
              "DICT_7X7_50": cv2.aruco.DICT_7X7_50, "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
              "DICT_7X7_250": cv2.aruco.DICT_7X7_250, "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
              "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
              "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
              "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
              "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
              "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11}

# Load the input image from disk and resize it
print("[INFO] Loading image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)

# Loop over the types of ArUCo dictionaries
for (arucoName, arucoDictionary) in ARUCO_DICT.items():
    # Load the ArUCo dictionary, grab the ArUCo parameters and attempt to detect the markers for the current dictionary
    arucoDict = cv2.aruco.Dictionary_get(arucoDictionary)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # If at least one ArUCo marker was detected display the ArUCo marker and its type name in the terminal
    if len(corners) > 0:
        print("[INFO] Detected {} markers for '{}'".format(len(corners), arucoName))
