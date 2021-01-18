# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)

# Specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100

# Define the path to the base output directory
BASE_OUTPUT = "output"

# Use the base output path to derive the path to the serialized model along with the training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "contrastive_plot.png"])

