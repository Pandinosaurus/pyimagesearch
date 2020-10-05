# ------------------------------------------------
#   USAGE
# ------------------------------------------------
# python predict.py --input output/test_images.txt

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input image/text file of image filenames")
args = vars(ap.parse_args())

# Determine the input file type, but assume default type as single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# If the file type is a text file, process *multiple* images
if "text/plain" == filetype:
    # Load the filenames in the testing file and initialize the list of image paths
    filenames = open(args["input"]).read().strip().split("\n")
    imagePaths = []
    # Loop over the filenames
    for f in filenames:
        # Construct the full path to the image filename and then update the image path list
        p = os.path.sep.join([config.IMAGES_PATH, f])
        imagePaths.append(p)

# Load the trained bounding box regression model from disk
print("[INFO] Loading object detector...")
model = load_model(config.MODEL_PATH)

# Loop over the image that are going to be used for testing the bounding box regression model
for imagePath in imagePaths:
    # Load the input image (in Keras format) from disk and preprocess it, scaling pixel intensities to the range [0,1]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # Make the bounding box predictions on the input image
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    # Load the input image (in OpenCV format), resize it such that it fits in the screen and grab its dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # Scale the predicted bounding box coordinates based on the image dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # Draw the predicted image bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


