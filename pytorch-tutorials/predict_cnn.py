# -----------------------------
#   USAGE
# -----------------------------
# python predict_cnn.py --model output/model.pth

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
import numpy as np
np.random.seed(42)      # Set the numpy seed for better reproducibility

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="Path to the trained PyTorch model")
args = vars(ap.parse_args())

# Set the device that will be used to test out the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the KMNIST dataset and randomly grab 10 data points from the dataset
print("[INFO] Loading the KMNIST test dataset...")
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)

# Initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)

# Load the model and set it to evaluation mode
model = torch.load(args["model"]).to(device)
model.eval()

# Switch off the autograd
with torch.no_grad():
    # Loop over the test set
    for (image, label) in testDataLoader:
        # Grab the original image and ground truth label
        origImage = image.numpy().squeeze(axis=(0, 1))
        gtLabel = testData.dataset.classes[label.numpy()[0]]
        # Send the input to the device and make predictions on it
        image = image.to(device)
        pred = model(image)
        # Find the class label index with the largest corresponding probability
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        predLabel = testData.dataset.classes[idx]
        # Convert the image from grayscale to RGB (in order to draw on the image)
        # and resize it (in order to easily see the image on the screen)
        origImage = np.dstack([origImage] * 3)
        origImage = imutils.resize(origImage, width=128)
        # Draw the predicted class label on it
        color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
        cv2.putText(origImage, gtLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
        # Display the result in terminal and show the input image
        print("[INFO] Ground Truth Label: {}, Predicted Label: {}".format(gtLabel, predLabel))
        cv2.imshow("Image", origImage)
        cv2.waitKey(0)

