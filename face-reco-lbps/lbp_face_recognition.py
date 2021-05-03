# -----------------------------
#   USAGE
# -----------------------------
# python lbp_face_recognition.py --input caltech_faces

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.faces import load_face_dataset
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="Path to input directory of images")
ap.add_argument("-f", "--face", type=str, default="face_detector", help="Path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the CALTECH face dataset
print("[INFO] Loading CALTECH dataset...")
(faces, labels) = load_face_dataset(args["input"], net, minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces)))

# Encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Construct the training and testing split
(trainX, testX, trainY, testY) = train_test_split(faces, labels, test_size=0.25, stratify=labels, random_state=42)

# Train the LBP face recognizer model
print("[INFO] Training the face recognizer model...")
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
start = time.time()
recognizer.train(trainX, trainY)
end = time.time()
print("[INFO] Training process took {:.4f} seconds".format(end - start))

# Initialize the list of predictions and confidence scores
print("[INFO] Gathering the predictions...")
predictions = []
confidence = []
start = time.time()

# Loop over the test data
for i in range(0, len(testX)):
    # Classify the face and update the list of predictions and confidence scores
    (prediction, conf) = recognizer.predict(testX[i])
    predictions.append(prediction)
    confidence.append(conf)

# Measure how long making predictions took
end = time.time()
print("[INFO] Inference process took {:.4f} seconds".format(end - start))

# Show the classification model
print(classification_report(testY, predictions, target_names=le.classes_))

# Generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)

# Loop over a sample of the testing data
for i in idxs:
    # Grab the predicted name and the actual name
    predName = le.inverse_transform([predictions[i]])[0]
    actualName = le.classes_[testY[i]]
    # Grab the face image and resize it in order to easily visualize the image on screen
    face = np.dstack([testX[i]] * 3)
    face = imutils.resize(face, width=250)
    # Draw the predicted name and actual name on the image
    cv2.putText(face, "Prediction: {}".format(predName), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(face, "Actual: {}".format(actualName), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Display the predicted name, actual name, and confidence of the prediction (i.e, chi-squared distance;
    # the *lower* the distance is the *more confident* the prediction is)
    print("[INFO] Prediction: {}, Actual: {}, Confidence: {:.2f}".format(predName, actualName, confidence[i]))
    # Display the current face to the screen
    cv2.imshow("Face", face)
    cv2.waitKey(0)





