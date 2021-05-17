# -----------------------------
#   USAGE
# -----------------------------
# python train_svr.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd

# Load the dataset, separate the features and the labels, and then perform a training and testing split using
# 85% of the data for training and 15% for evaluation
print("[INFO] Loading dataset...")
dataset = pd.read_csv(config.CSV_PATH, names=config.COLS)
dataX = dataset[dataset.columns[:-1]]
dataY = dataset[dataset.columns[-1]]
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, random_state=3, test_size=0.15)

# Standardize the feature values by computing the mean, subtracting the mean from the data points and then dividing by
# the standard deviation
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

# Train the model with *no* hyperparameter tuning
print("[INFO] Training the support vector regression model...")
model = SVR()
model.fit(trainX, trainY)

# Evaluate the model using R^2-score (1.0 is the best value)
print("[INFO] Evaluating the model...")
print("R2: {:.2f}".format(model.score(testX, testY)))

