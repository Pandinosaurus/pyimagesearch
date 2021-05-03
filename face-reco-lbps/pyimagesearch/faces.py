# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from imutils import paths
import numpy as np
import cv2
import os


# -----------------------------
#   FUNCTIONS
# -----------------------------
def detect_faces(net, image, minConfidence=0.5):
    # Grab the dimensions of the image and then construct a blob from the image
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Pass the blob through the network to obtain the face detections,
    # then initialize a list to store the predicted bounding boxes
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e, probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > minConfidence:
            # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Update the bounding box results list
            boxes.append((startX, startY, endX, endY))
    # Return the face detection bounding boxes
    return boxes


def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=15):
    # Grab the paths to all images in the input directory, extract the name of the person (i.e, class label) from the
    # directory structure, and count the number of example images per face
    imagePaths = list(paths.list_images(inputPath))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()
    # Initialize lists to store the extracted faces and associated labels
    faces = []
    labels = []
    # Loop over the image paths
    for imagePath in imagePaths:
        # Load the image from disk and extract the name of the person from the subdirectory structure
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]
        # Only process images that have a sufficient number of examples belonging to the class
        if counts[names.index(name)] < minSamples:
            continue
        # Perform face detection
        boxes = detect_faces(net=net, image=image, minConfidence=minConfidence)
        # Loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # Extract the face ROI, resize it and then convert it to grayscale
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            # Update the faces and labels lists
            faces.append(faceROI)
            labels.append(name)
    # Convert the faces and labels list to NumPy arrays
    faces = np.array(faces)
    labels = np.array(labels)
    # Return a 2-tuple of the faces and labels
    return faces, labels


