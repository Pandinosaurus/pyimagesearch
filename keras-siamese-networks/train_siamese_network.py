# -------------------------------
#   USAGE
# -------------------------------
# python train_siamese_network.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.siamese_network import build_siamese_model
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] Loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# Prepare the positive and negative pairs
print("[INFO] Preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

# Configure the siamese network
print("[INFO] Building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# Finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# Compile the model
print("[INFO] Compiling the model...")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
print("[INFO] Training the model...")
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
                    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
                    batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)

# Serialize the model to disk
print("[INFO] Saving siamese model...")
model.save(config.MODEL_PATH)

# Plot the training history
print("[INFO] Plotting training history...")
utils.plot_training(history, config.PLOT_PATH)
