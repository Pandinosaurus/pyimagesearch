# -----------------------------------------------
#   USAGE
# -----------------------------------------------
# python test_siamese_network.py --input examples

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to input directory of testing images")
args = vars(ap.parse_args())

# Grab the test dataset image paths and then randomly generate a total of 10 image pairs
print("[INFO] Loading test dataset...")
testImagePaths = list(list_images(args["input"]))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))

# Load the model from disk
print("[INFO] Loading siamese network model...")
model = load_model(config.MODEL_PATH)

# Loop over all image pairs
for (i, (pathA, pathB)) in enumerate(pairs):
    # Load both the images and convert them to grayscale
    imageA = cv2.imread(pathA, 0)
    imageB = cv2.imread(pathB, 0)
    # Create a copy of both the images for visualization purpose
    origA = imageA.copy()
    origB = imageB.copy()
    # Add channel a dimension to both the images
    imageA = np.expand_dims(imageA, axis=-1)
    imageB = np.expand_dims(imageB, axis=-1)
    # Add a batch dimension to both images
    imageA = np.expand_dims(imageA, axis=0)
    imageB = np.expand_dims(imageB, axis=0)
    # Scale the pixel values to the range of [0, 1]
    imageA = imageA / 255.0
    imageB = imageB / 255.0
    # Use the siamese network model to make predictions on the image pair,
    # indicating whether or not the images belong to the same class
    preds = model.predict([imageA, imageB])
    proba = preds[0][0]
    # Initialize the figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    plt.suptitle("Similarity: {:.2f}".format(proba))
    # Show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")
    # Show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")
    # Show the plot
    plt.show()

