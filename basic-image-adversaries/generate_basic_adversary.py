# -----------------------------
#   USAGE
# -----------------------------
# python generate_basic_adversary.py --input pig.jpg --output adversarial.png --class-idx 341

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def preprocess_image(image):
    """
        Swaps color channels, resizes the input image and adds a batch dimension
        :param image: input image
        :return: preprocessed image
    """
    # Swap color channels, resize the input image and add a batch dimension
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    # Return the preprocessed image
    return image


def clip_eps(tensor, eps):
    """
        Clips the values of the tensor according to a given range and returns it
        :param tensor: list of tensors
        :param eps: epochs range
        :return: tensor value according to given range
    """
    # Clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor, clip_value_min=-eps, clip_value_max=eps)


def generate_adversaries(model, baseImage, delta, classIdx, steps=50):
    """
        Generate binaries adversaries for the input image given a base model
        :param model: base model
        :param baseImage: input image
        :param delta: perturbation vector
        :param classIdx: original class index
        :param steps: number of steps
        :return: new perturbation vector
    """
    # Iterate over the number of steps
    for step in range(0, steps):
        # Record the gradients
        with tf.GradientTape() as tape:
            # Explicitly indicate that the perturbation vector should be tracked for gradient updates
            tape.watch(delta)
            # Add the perturbation vector to the base image and preprocess the resulting image
            adversary = preprocess_input(baseImage + delta)
            # Run this newly constructed image tensor through the base model and calculate the loss with respect to the
            # original class index
            predictions = model(adversary, training=False)
            loss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions)
            # Check the loss value and display it in the terminal
            if step % 5 == 0:
                print("[INFO] Step: {}, Loss: {}...".format(step, loss.numpy()))
        # Calculate the gradients of loss with respect to the perturbation vector
        gradients = tape.gradient(loss, delta)
        # Update the weights, clip the perturbation vector and update its value
        optimizer.apply_gradients([(gradients, delta)])
        delta.assign_add(clip_eps(delta, eps=EPS))
    # Return the perturbation vector
    return delta


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to original input image")
ap.add_argument("-o", "--output", required=True, help="path to output adversarial image")
ap.add_argument("-c", "--class-idx", type=int, required=True, help="ImageNet class ID of the predicted label")
args = vars(ap.parse_args())

# Define the epsilon and learning rate constants
EPS = 2 / 255.0
LR = 0.1

# Load the input image from disk and preprocess it
print("[INFO] Loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)

# Load the pre-trained ResNet50 model for running inference
print("[INFO] Loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

# Initialize the optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()

# Create a tensor based of the input image and initialize the perturbation vector that will be updated via training
baseImage = tf.constant(image, dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

# Generate the perturbation vector to create the adversarial example of the base image
print("[INFO] Generating perturbation...")
deltaUpdated = generate_adversaries(model, baseImage, delta, args["class_idx"])

# Create the adversarial example, swap color channels and save the output image to disk
print("[INFO] Creating adversarial example of the base image...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage, 0, 255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"], adverImage)

# Run inference with this adversarial example, parse the results and display the top-1 predicted results
print("[INFO] Running inference on the adversarial example...")
preprocessedImage = preprocess_input(baseImage + deltaUpdated)
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions, top=3)[0]
label = predictions[0][1]
confidence = predictions[0][2] * 100
print("[INFO] Label: {} Confidence: {:.2f}%".format(label, confidence))

# Draw the top-most predicted label on the adversarial image along with the confidence score
text = "{}: {:.2f}%".format(label, confidence)
cv2.putText(adverImage, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Output", adverImage)
cv2.waitKey(0)