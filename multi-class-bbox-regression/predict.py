# -----------------------------------------------
#   USAGE
# -----------------------------------------------
# python predict.py --input output/test_paths.txt

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
import pickle
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input image/text file of image paths")
args = vars(ap.parse_args())

# Determine the input file type, but assume default type as single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# If the file type is a text file, then process *multiple* images
if "text/plain" == filetype:
    # Load the image paths in the testing file
    imagePaths = open(args["input"]).read().strip().split("\n")

# Load the object detector and the label binarizer from disk
print("[INFO] Loading object detector...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

# Loop over the image that are going to be used for testing the bounding box regression model
for imagePath in imagePaths:
    # Load the input image (in keras format) from disk and preprocess it, scaling the pixel intensities to range [0, 1]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # Predict the bounding box of the object along with the class label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    # Determine the class label with the largest predicted probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]
    # Load the input image (in OpenCV format), resize it such that it fits the screen and grab its dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # Scale the predicted bounding box coordinates based on the image dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # Draw the predicted bounding boxes and class labels on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)





