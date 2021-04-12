# -----------------------------
#   USAGE
# -----------------------------
# python video_face_detector.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str, default="cascades/haarcascade_frontalface_default.xml",
                help="Path to haar cascade face detector .xml file")
args = vars(ap.parse_args())

# Load the Haar Cascade Face Detector from disk
print("[INFO] Loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the image from the video stream, resize it and then convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    print("[INFO] Performing face detection...")
    rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    print("[INFO] {} faces detected!".format(len(rects)))
    # Loop over the bounding boxes
    for (x, y, w, h) in rects:
        # Draw the face bounding box on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show the output image frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the 'Q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

