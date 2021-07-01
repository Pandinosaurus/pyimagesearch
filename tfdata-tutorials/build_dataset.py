# -----------------------------
#   USAGE
# -----------------------------
# python build_dataset.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from imutils import paths
import random
import shutil
import os

# Grab the paths to all the input images in the original input directory and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# Compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# Use part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# Define the datasets that are going to be built
datasets = [("training", trainPaths, config.TRAIN_PATH),
            ("validation", valPaths, config.VAL_PATH),
            ("testing", testPaths, config.TEST_PATH)]

# Loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    # Show data split
    print("[INFO] Building '{}' split".format(dType))
    # If the output base output directory does not exist, then create it
    if not os.path.exists(baseOutput):
        print("[INFO] Creating '{}' directory".format(baseOutput))
        os.makedirs(baseOutput)
    # Loop over the input image paths
    for inputPath in imagePaths:
        # Extract the filename of the input image and extract the class label
        # ("0" for "negative" and "1" for "positive")
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]
        # Build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])
        # If the label output directory does not exit, create it
        if not os.path.exists(labelPath):
            print("[INFO] Creating '{}' directory".format(labelPath))
            os.makedirs(labelPath)
        # Construct the path to the destination image and then copy the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)

