# -----------------------------
#   USAGE
# -----------------------------
# python super_res_video.py --model models/ESPCN_x4.pb

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path to super resolution model")
args = vars(ap.parse_args())

# Extract the model name and the model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# Initialize the OpenCV's super resolution DNN object, load the super resolution model from disk and set the model name
# and the model scale
print("[INFO] Loading Super Resolution Model: {}".format(args["model"]))
print("[INFO] Model Name: {}".format(modelName))
print("[INFO] Model Scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting the video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frames from the threaded video stream and resize it to have a maximum width of 300 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=300)
    # Upscale the frame using the super resolution model and then bicubic interpolation (in order to visualize and
    # compare the two images)
    upscaled = sr.upsample(frame)
    bicubic = cv2.resize(frame, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC)
    # Show the original frame, bicubic interpolation frame and super resolution frame
    cv2.imshow("Original", frame)
    cv2.imshow("Bicubic", bicubic)
    cv2.imshow("Super Resolution", upscaled)
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


