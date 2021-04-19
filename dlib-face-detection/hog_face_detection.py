# -----------------------------------
#   USAGE
# -----------------------------------
# python hog_face_detection.py --image images/family.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to the input image file")
ap.add_argument("-u", "--upsample", type=int, default=1, help="Number of times to upsample")
args = vars(ap.parse_args())

# Load DLIB's HOG + Liner SVM Face Detector model
print("[INFO] Loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()

# Load the input image from disk, resize it and then convert it from BGR to RGB channel ordering
# (which is what dlib expects)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform face detection using dlib's frontal face detector
start = time.time()
print("[INFO] Performing face detection with dlib...")
rects = detector(rgb, args["upsample"])
end = time.time()
print("[INFO] Face detection took {:.4f} seconds".format(end - start))

# Convert the resulting dlib rectangle objects to bounding boxes, then ensure the bounding boxes are all within the
# bounds of the input image
boxes = [convert_and_trim_bb(image, r) for r in rects]

# Loop over the bounding boxes
for (x, y, w, h) in boxes:
    # Draw the bounding box on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)


