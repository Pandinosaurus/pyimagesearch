# -----------------------------
#   USAGE
# -----------------------------
# python train_tf_scikit.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import tensorflow as tf
tf.random.set_seed(42)  # Import tensorflow and fix the random seed for better reproducibility
from pyimagesearch.mlp import get_mlp_model
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
print("[INFO] Downloading the MNIST dataset...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Scale data to the range [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Initialize the model with the default hyperparameter values
print("[INFO] Initializing the model...")
model = get_mlp_model()

# Train the network (i.e, with no hyperparameter tuning)
print("[INFO] Training the model...")
H = model.fit(x=trainData, y=trainLabels, validation_data=(testData, testLabels), batch_size=8, epochs=20)

# Make predictions on the test set and evaluate it
print("[INFO] Evaluating the model...")
accuracy = model.evaluate(testData, testLabels)[1]
print("Accuracy: {:.2f}%".format(accuracy * 100))
