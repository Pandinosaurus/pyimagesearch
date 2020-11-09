# -----------------------------
#   USAGE
# -----------------------------
# python super_res_image.py --model models/LapSRN_x8.pb --image examples/zebra.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import argparse
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to super resolution model")
ap.add_argument("-i", "--image", required=True, help="Path to input image we want to increase resolution of")
args = vars(ap.parse_args())

# Extract the model name and model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# Initialize OpenCV's super resolution DNN object, load the super model from disk and set the model name and scale
print("[INFO] Loading Super Resolution Model: {}".format(args["model"]))
print("[INFO] Model Name: {}".format(modelName))
print("[INFO] Model Scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

# Load the input image from disk and display its spatial dimensions
image = cv2.imread(args["image"])
print("[INFO] Width: {}, Height: {}".format(image.shape[1], image.shape[0]))

# Use the super resolution model to upscale the image, timing how long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] Super resolution took {:.6f} seconds".format(end - start))

# Show the spatial dimensions of the super resolution image
print("[INFO] Width: {}, Height: {}".format(upscaled.shape[1], upscaled.shape[0]))

# Resize the image using standard bicubic interpolation
start = time.time()
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)
end = time.time()
print("[INFO] Bicubic interpolation took {:.6f} seconds".format(end - start))

# Show the original input image, bicubic interpolation image and the super resolution deep learning output
cv2.imshow("Original", image)
cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)
cv2.waitKey(0)

