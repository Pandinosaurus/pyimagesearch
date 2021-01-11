# -----------------------------
#   USAGE
# -----------------------------
# python opencv_ar_video.py --input sources/jp_trailer_short.mp4

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.augmented_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="Path to input video file for augmented reality")
ap.add_argument("-c", "--cache", type=int, default=-1, help="Whether or not to use reference points cache")
args = vars(ap.parse_args())

# Load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] Initializing the marker detector...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()

# Initialize the video file stream
print("[INFO] Accessing the video stream...")
vf = cv2.VideoCapture(args["input"])

# Initialize a queue to maintain the next frame from the video file stream
Q = deque(maxlen=128)

# In order to start the augmented reality, first we need to save a frame in the queue and so read the next frame
# from the video file source and it to the queue
(grabbed, source) = vf.read()
Q.appendleft(source)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting the video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while len(Q) > 0:
    # Grab the frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    # Attempt to find the ArUCo markers in the frame and provide them in the image if found,
    # take the current source image and warp it onto the input frame using the augmented reality technique
    warped = find_and_warp(frame, source, cornerIDs=(923, 1001, 241, 1007), arucoDict=arucoDict,
                           arucoParams=arucoParams, useCache=args["cache"] > 0)
    # If the warped frame is not None, then it is possible to assume that:
    # (1) The four markers have been found
    # (2) The perspective warp was successfully applied
    if warped is not None:
        # Set the frame to the output augmented reality frame and then grab the next video file frame from the queue
        frame = warped
        source = Q.popleft()
    # For speed/efficiency, it is possible to use the queue to keep the next video frame queue ready and accessible
    # -- the trick is to ensure the queue is always (or nearly) full
    if len(Q) != Q.maxlen:
        # Read the next frame from the video file stream
        (grabbed, nextFrame) = vf.read()
        # If the frame was read (meaning it is not at the end of the video file stream), add the frame to the queue
        if grabbed:
            Q.append(nextFrame)
    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
