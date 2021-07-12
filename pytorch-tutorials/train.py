# -----------------------------
#   USAGE
# -----------------------------
# python train.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch


# -----------------------------
#   FUNCTIONS
# -----------------------------
def next_batch(inputs, targets, batchSize):
    # Loop over the dataset
    for i in range(0, inputs.shape[0], batchSize):
        # Yield a tuple of the current batched data and labels
        yield inputs[i:i + batchSize], targets[i:i + batchSize]


# Specify the batch size, number of epochs and the learning rate
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

# Determine the device that is going to be used for training the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Training using {}...".format(DEVICE))

# Generate a 3-class classification problem with 1000 data points, where each data point is a 40 feature vector
print("[INFO] Preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3, cluster_std=2.5, random_state=95)

# Creating training and testing data splits and convert them to PyTorch tensors
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.15, random_state=95)
# Convert from NumPy arrays to PyTorch tensors
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

# Initialize the model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print("[INFO] Model Architecture: {}".format(mlp))

# Initialize the optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

# Create a template to summarize the current training process
trainTemplate = "Epoch: {} Test Loss: {:.3f} Test Accuracy: {:.3f}"

# Loop through the epochs
for epoch in range(0, EPOCHS):
    # Initialize the tracker variables and set the model to trainable
    print("[INFO] Epoch: {}...".format(epoch + 1))
    trainLoss = 0
    trainAcc = 0
    samples = 0
    mlp.train()
    # Loop over the current batch of data
    for (batchX, batchY) in next_batch(inputs=trainX, targets=trainY, batchSize=BATCH_SIZE):
        # Flash data to the current device, run it through the model and calculate the loss
        (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
        predictions = mlp(batchX)
        loss = lossFunc(predictions, batchY.long())
        # Zero the gradients accumulated from the previous steps, perform backpropagation
        # and update the current model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Update training loss, accuracy and the number of samples visited
        trainLoss += loss.item() * batchY.size(0)
        trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        samples += batchY.size(0)
    # Display the model progress on the current training data batch
    trainTemplate = "Epoch: {} Train Loss: {:.3f} Train Accuracy: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples), (trainAcc / samples)))
    # Initialize the tracker variables for testing and set the model to evaluation mode
    testLoss = 0
    testAcc = 0
    samples = 0
    mlp.eval()
    # Initialize the non-gradient context
    with torch.no_grad():
        # Loop over the current batch of test data
        for (batchX, batchY) in next_batch(inputs=testX, targets=testY, batchSize=BATCH_SIZE):
            # Flash the data to the current device
            (batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
            # Run data through the model and calculate the loss
            predictions = mlp(batchX)
            loss = lossFunc(predictions, batchY.long())
            # Update the test loss, accuracy and the number of samples visited
            testLoss += loss.item() * batchY.size(0)
            testAcc += (predictions.max(1)[1] == batchY).sum().item()
            samples += batchY.size(0)
        # Display the model progress on the current test data batch
        testTemplate = "Epoch: {} Test Loss: {:.3f} Test Accuracy: {:.3f}"
        print(testTemplate.format(epoch + 1, (testLoss / samples), (testAcc / samples)))
        print("")




