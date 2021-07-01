# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# Initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = os.path.join("datasets", "orig")

# Initialize the base path to the *new* directory that will contain the images
# after computing the training and testing splits
BASE_PATH = os.path.join("datasets", "idc")

# Derive the training, validation and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# Define the amount of data that will be used for training
TRAIN_SPLIT = 0.8

# Define the amount of validation data that will be a percentage of the *training* data
VAL_SPLIT = 0.1

# Define the input image spatial dimensions
IMAGE_SIZE = (48, 48)

# Initialize the number of epochs, early stopping patience, initial learning rate and the batch size
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-2
BS = 128
