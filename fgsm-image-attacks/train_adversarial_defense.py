# -----------------------------------
#   USAGE
# -----------------------------------
# python train_adversarial_defense.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
from pyimagesearch.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np


# Load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] Loading the MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# One-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Initialize the optimizer and model
print("[INFO] Compiling the model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the simple CNN on MNIST
print("[INFO] Training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=20, verbose=1)

# Make predictions on the testing set for the model trained on non-adversarial images
(loss, accuracy) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] Normal testing images:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))

# Generate a set of adversarial from our test set
print("[INFO] Generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(testX), testX, testY, (28, 28, 1), eps=0.1))

# Re-evaluate the model on the adversarial images
(loss, accuracy) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] Adversarial testing images:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))

# Lower the learning rate and re-compile the model (in order to fine tune the model on the adversarial images)
print("[INFO] Re-compiling model...")
opt = Adam(lr=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Fine-tune the CNN on the adversarial images
print("[INFO] Fine-tuning network on adversarial examples...")
model.fit(advX, advY, batch_size=64, epochs=10, verbose=1)

# Now that the model is fine-tuned, evaluate it on the test set (i.e., non-adversarial)
# again to see if the overall performance has degraded
(loss, accuracy) = model.evaluate(x=testX, y=testY, verbose=0)
print("")
print("[INFO] Normal testing images *after* fine-tuning:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}\n".format(loss, accuracy))

# Do a final evaluation of the model on the adversarial images
(loss, accuracy) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] Adversarial images *after* fine-tuning:")
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}".format(loss, accuracy))

