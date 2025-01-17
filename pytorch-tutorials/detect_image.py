# -----------------------------
#   USAGE
# -----------------------------
# python detect_image.py --model frcnn-resnet --image data/examples/example_01.jpg --labels data/labels/coco_classes.pickle

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Path to the input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
                choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet", "maskrcnn"],
                help="Name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
                help="Path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
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
model = MODELS[args["model"]](pretrained=True, progress=True,
                              num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

# Load the image file from disk
image = cv2.imread(args["image"])
orig = image.copy()

# Convert the image from BGR to RGB channel ordering (OpenCV)
# and change the image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

# Add the batch dimension, scale the raw pixel intensities to the range [0, 1]
# and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

# Send the input to the device and pass the input through the network to get the detection and predictions
image = image.to(DEVICE)
detections = model(image)[0]

# Loop over the detections
for i in range(0, len(detections["boxes"])):
    # Extract the confidence (i.e, probability) associated with the prediction
    confidence = detections["scores"][i]
    # Filter out weak detection by ensuring the confidence is greater than the minimum confidence
    if confidence > args["confidence"]:
        # Extract the index of the class label from the detections,
        # then compute the (x,y) coordinates of the bounding box for the object
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype("int")
        # Display the prediction to the terminal
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))
        # Draw the bounding box and label on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)



