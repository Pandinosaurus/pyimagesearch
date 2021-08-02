# -----------------------------
#   USAGE
# -----------------------------
# python detect_realtime.py --model frcnn-resnet --labels data/labels/coco_classes.pickle

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
                choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet", "maskrcnn"],
                help="Name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
                help="Path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Set the device that is going to be used to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the list of categories in the COCO dataset and then generate a set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize a dictionary containing the model name and its corresponding torchvision function call
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
    "maskrcnn": detection.maskrcnn_resnet50_fpn
}

# Load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(
    DEVICE)
model.eval()

# Initialize the video stream, allow the camera sensor to warm up and initialize the FPS counter
print("[INFO] Starting the video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()
    # Convert the image from BGR to RGB channel ordering (OpenCV)
    # and change the image from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    # Add a batch dimension, scale the raw pixel intensities to the range [0, 1]
    # and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    # Send the input to the device and pass the input through the network to get the detections and predictions
    frame = frame.to(DEVICE)
    detections = model(frame)[0]
    # Loop over the detections
    for i in range(0, len(detections["boxes"])):
        # Extract the confidence (i.e, probability) associated with the prediction
        confidence = detections["scores"][i]
        # Filter out the weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            # Extract the index of the class label from the detections,
            # then compute the (x, y) coordinates of  the bounding box for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # Draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    # Show the output frame
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
    # Update the FPS counter
    fps.update()

# Stop the timer and display the FPS information
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
