# -----------------------------
#   USAGE
# -----------------------------
# python read_from_disk.py --dataset ../datasets/fruits

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import os


# -----------------------------
#   FUNCTIONS
# -----------------------------
def load_images(imagePath):
    # Read the image data from disk, decode it, resize it and scale the pixel intensities to the range [0, 1]
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (96, 96)) / 255.0
    # Grab the label and encode it
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    oneHot = label == classNames
    encodedLabel = tf.argmax(oneHot)
    # Return the image and the integer encoded label
    return image, encodedLabel


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the input dataset")
args = vars(ap.parse_args())

# Initialize the batch size and the number of steps
BS = 64
NUM_STEPS = 1000

# Grab the list of images in the dataset directory and grab all the unique class names
print("[INFO] Loading image paths...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = np.array(sorted(os.listdir(args["dataset"])))

# Build the dataset and data input pipeline
print("[INFO] Creating a tf.data input pipeline")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = (
    dataset.shuffle(1024).map(load_images, num_parallel_calls=AUTOTUNE).cache().repeat().batch(BS).prefetch(AUTOTUNE)
)

# Create a standard image generator object
print("[INFO] Creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator(rescale=1.0/255)
dataGen = imageGen.flow_from_directory(args["dataset"], target_size=(96, 96), batch_size=BS, class_mode="categorical",
                                       color_mode="rgb")

# Benchmark the image data generator and display the number of data points generated,
# along with the time taken to perform the operation
total_time = benchmark(datasetGen=dataGen, numSteps=NUM_STEPS)
print("[INFO] ImageDataGenerator generated {} images in  {:.2f} seconds...".format(BS * NUM_STEPS, total_time))

# Create a dataset iterator, benchmark the tf.data.pipeline and display the number of data points generated,
# along with the time taken to perfrom the operation
datasetGen = iter(dataset)
total_time = benchmark(datasetGen, NUM_STEPS)
print("[INFO] tf.data generated {} images in {:.2f} seconds...".format(BS * NUM_STEPS, total_time))
