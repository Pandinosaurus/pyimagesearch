# -----------------------------
#   USAGE
# -----------------------------
# python train_cnn.py --model output/model.pth --plot output/plot.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="Path to the output trained model")
ap.add_argument("-p", "--plot", type=str, required=True, help="Path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Define the training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# Define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# Set the device that will be used for training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the KMNIST dataset
print("[INFO] Loading the KMNIST dataset...")
trainData = KMNIST(root="data", train=True, download=True, transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True, transform=ToTensor())

# Calculate the train/validation data splits
print("[INFO] Generating the train/validation data splits...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData, [numTrainSamples, numValSamples],
                                    generator=torch.Generator().manual_seed(42))

# Initialize the train, validation and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# Calculate the steps per epoch for the training and validation sets
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# Initialize the LeNet model
print("[INFO] Initializing the LeNet model...")
model = LeNet(numChannels=1, classes=len(trainData.dataset.classes)).to(device)

# Initialize the optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# Initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# Measure how long the training process is going to take
print("[INFO] Training the network...")
startTime = time.time()

# Loop over the epochs
for e in range(0, EPOCHS):
    # Set the model in the training mode
    model.train()
    # Initialize the total training and validation loss variables
    totalTrainLoss = 0
    totalValLoss = 0
    # Initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0
    # Loop over the training set
    for (x, y) in trainDataLoader:
        # Send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # Perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # Zero out the gradients, perform backpropagation and update the current weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Add the loss to the total training loss so far and calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Switch off the autograd for evaluation
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        # Loop over the validation data set
        for (x, y) in valDataLoader:
            # Send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # Make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
            # Calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Calculate the average training loss and the average validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
    # Calculate the training and the validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
    # Update the training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
    # Print the model training and the validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train Loss: {:.6f}, Train Accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Validation Loss: {:.6f}, Validation Accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

# Finish measuring how long the training process took
endTime = time.time()
print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime - startTime))

# Evaluate the network on the test set
print("[INFO] Evaluating the network...")
# Turn off the autograd for testing evaluation
with torch.no_grad():
    # Set the model to evaluation mode
    model.eval()
    # Initialize a list to store the predictions
    preds = []
    # Loop over the test set
    for (x, y) in testDataLoader:
        # Send the input to the device
        x = x.to(device)
        # Make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# Generate a classification report
print("[INFO] Classification Report: ")
print(classification_report(testData.targets.cpu().numpy(), np.array(preds), target_names=testData.classes))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Serialize the model to disk
torch.save(model, args["model"])

