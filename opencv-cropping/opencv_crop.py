# -----------------------------
#   USAGE
# -----------------------------
# python opencv_crop.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="adrian.png", help="Path to the input image")
args = vars(ap.parse_args())

# Load the input image and display it to the screen
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Cropping an image with OpenCV is accomplished via simple NumPy array slices in startY:endY, startX:endX order
# Here we are cropping the face from the image(these coordinates were determined using photo editing software such
# as PhotoShop, GIMP, Paint, etc...)
face = image[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)

# Apply another image crop, this time extracting the body
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)


