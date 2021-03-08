# ----------------------------------
#   USAGE
# ----------------------------------
# python train_adversarial_attack.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
from pyimagesearch.fgsm import generate_image_adversary
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

# Load the MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] Loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# One hot encode the labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Initialize the optimizer and the model
print("[INFO] Compiling the model...")
opt = Adam(lr=1e-3)
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the simple CNN on the MNIST dataset
print("[INFO] Training the network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=10, verbose=1)

# Make predictions on the testing set for the model trained on non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] Loss: {:.4f}, Accuracy: {:.4f}".format(loss, acc))

# Loop over a sample of the testing images
for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
    # Grab the current image and label
    image = testX[i]
    label = testY[i]
    # Generate an image adversary for the current image and make a prediction on the adversary image
    adversary = generate_image_adversary(model, image.reshape(1, 28, 28, 1), label, eps=0.1)
    pred = model.predict(adversary)
    # Scale both the original image and the adversary image to the range [0, 255]
    # and convert them to unsigned 8-bit integers
    adversary = adversary.reshape((28, 28)) * 255
    adversary = np.clip(adversary, 0, 255).astype("uint8")
    image = image.reshape((28, 28)) * 255
    image = image.astype("uint8")
    # Convert the image and adversarial image from grayscale to three channel (in order to draw on the image)
    image = np.dstack([image] * 3)
    adversary = np.dstack([adversary] * 3)
    # Resize the images in order to visualize them later
    image = cv2.resize(image, (96, 96))
    adversary = cv2.resize(adversary, (96, 96))
    # Determine the predicted label for both the original image and the adversarial image
    imagePred = label.argmax()
    adversaryPred = pred[0].argmax()
    color = (0, 255, 0)
    # If the image prediction does not match with the adversarial prediction then update the color
    if imagePred != adversaryPred:
        color = (0, 0, 255)
    # Draw the predictions on the respective output images
    cv2.putText(image, str(imagePred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)
    cv2.putText(adversary, str(adversaryPred), (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    # Stack the two images horizontally and then show the original image and its adversary
    output = np.hstack([image, adversary])
    cv2.imshow("FGSM Adversarial Images", output)
    cv2.waitKey(0)


