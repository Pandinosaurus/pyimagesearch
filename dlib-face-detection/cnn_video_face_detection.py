# -----------------------------------
#   USAGE
# -----------------------------------
# python cnn_video_face_detection.py --image images/concert.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
from imutils.video import VideoStream
import argparse
import imutils
import time
import dlib
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="mmod_human_face_detector.dat",
                help="Path to dlib's CNN face detector model file")
ap.add_argument("-u", "--upsample", type=int, default=1, help="Number of times to upsample")
args = vars(ap.parse_args())

# Load DLIB'S CNN Face Detector model
print("[INFO] Loading CNN face detection model...")
detector = dlib.cnn_face_detection_model_v1(args["model"])

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream, resize it to have a maximum width of 600 pixels and then convert it
    # from BGR to RGB channel ordering (which is what the dlib expects)
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform face detection using dlib's face detector
    start = time.time()
    print("[INFO] Performing face detection with dlib face detector...")
    results = detector(rgb, args["upsample"])
    end = time.time()
    print("[INFO] Face detection took {:.4f} seconds".format(end - start))
    # Convert the resulting dlib rectangle objects to bounding boxes, then ensure that the bounding boxes are all within
    # the bounds of the input image
    boxes = [convert_and_trim_bb(frame, r.rect) for r in results]
    # Loop over the bounding boxes
    for (x, y, w, h) in boxes:
        # Draw the bounding box on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Show the output image
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()




