# -----------------------------
#   USAGE
# -----------------------------
# python train_model.py --dataset texture_dataset

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import argparse
import time
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the input dataset")
args = vars(ap.parse_args())

# Grab the image paths in the input dataset directory
imagePaths = list(paths.list_images(args["dataset"]))

# Initialize the local binary patterns descriptor along with the data and label lists
print("[INFO] Extracting features...")
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# Loop over the dataset of images
for imagePath in imagePaths:
    # Load the image, convert it to grayscale and then quantify it using LBPs
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # Extract the label from the image path, then update the label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

# Partition the data into training and testing splits using 75% for training and the remaining 25% for testing
print("[INFO] Constructing training/testing split...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, random_state=22, test_size=0.25)

# Construct the set of hyperparameters to tune
parameters = [
    {"kernel": ["linear"], "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
    {"kernel": ["poly"], "degree": [2, 3, 4], "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
    {"kernel": ["rbf"], "gamma": ["auto", "scale"], "C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]

# Tune the hyperparameters via cross-validation grid search
print("[INFO] Tuning hyperparameters via grid search cross-validation...")
grid = GridSearchCV(estimator=SVC(), param_grid=parameters, n_jobs=-1)
start = time.time()
grid.fit(trainX, trainY)
end = time.time()

# Show the grid search information
print("[INFO] Grid search took {:.2f} seconds".format(end - start))
print("[INFO] Grid search best score: {:.2f}%".format(grid.best_score_ * 100))
print("[INFO] Grid search best parameters: {}".format(grid.best_params_))

# Grab the best model and evaluate it
print("[INFO] Evaluating the model...")
model = grid.best_estimator_
predictions = model.predict(testX)
print(classification_report(testY, predictions))


