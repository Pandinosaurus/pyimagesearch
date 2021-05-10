# -----------------------------
#   USAGE
# -----------------------------
# python eigenfaces.py --input caltech_faces --visualize 1

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from pyimagesearch.faces import load_face_dataset
from imutils import build_montages
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
ap.add_argument("-n", "--num-components", type=int, default=150, help="Number of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="Whether or not PCA components should be visualized")
args = vars(ap.parse_args())

# Load the serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the CALTECH faces dataset
print("[INFO] Loading dataset...")
(faces, labels) = load_face_dataset(inputPath=args["input"], net=net, minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces)))

# Flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])

# Encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Construct the training and testing splits
split = train_test_split(faces, pcaFaces, labels, test_size=0.25, stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split

# Compute the PCA (eigenfaces) representation of the data, then project te training data onto the eigenfaces subspace
print("[INFO] Creating eigenfaces...")
pca = PCA(svd_solver="randomized", n_components=args["num_components"], whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] Computing eigenfaces took {:.4f} seconds".format(end - start))

# Check to see if the PCA components should be visualized
if args["visualize"] > 0:
    # Initialize the list of images in the montage
    images = []
    # Loop over the first 16 individual components
    for (i, component) in enumerate(pca.components_[:16]):
        # Reshape the component to a 2D matrix, then convert the data type to an unsigned 8-bit integer
        # so it can be displayed with OpenCV
        component = component.reshape((62, 47))
        component = rescale_intensity(component, out_range=(0, 255))
        component = np.dstack([component.astype("uint8")] * 3)
        images.append(component)
    # Construct the montage for the images
    montage = build_montages(images, (47, 62), (4, 4))[0]
    # Show the mean and principal component visualizations with the mean image
    mean = pca.mean_.reshape((62, 47))
    mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    cv2.imshow("Mean", mean)
    cv2.imshow("Components", montage)
    cv2.waitKey(0)

# Train a classifier on the eigenfaces representation
print("[INFO] Training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)

# Evaluate the model
print("[INFO] Evaluating the model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions, target_names=le.classes_))

# Generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)

# Loop over a sample of the testing data
for i in idxs:
    # Grab the predicted name and actual name
    predName = le.inverse_transform([predictions[i]])[0]
    actualName = le.classes_[testY[i]]
    # Grab the face image and resize it such that it is possible to easily see it on screen
    face = np.dstack([origTest[i]] * 3)
    face = imutils.resize(face, width=250)
    # Draw the predicted name and actual name on the image
    cv2.putText(face, "Prediction: {}".format(predName), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(face, "Actual: {}".format(actualName), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Display the predicted name and actual name
    print("[INFO] Prediction: {}, Actual: {}".format(predName, actualName))
    # Display the current face to the screen
    cv2.imshow("Face", face)
    cv2.waitKey(0)

