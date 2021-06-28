# -----------------------------
#   USAGE
# -----------------------------
# python read_from_memory.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100
from tensorflow.data import AUTOTUNE
import tensorflow as tf

# Initialize the batch size and number of steps
BS = 64
NUM_STEPS = 5000

# Load the CIFAR-100 dataset from keras datasets
print("[INFO] Loading the CIFAR-100 dataset from keras...")
((trainX, trainY), (testX, testY)) = cifar100.load_data()

# Create a standard image generator object
print("[INFO] Creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator()
dataGen = imageGen.flow(x=trainX, y=trainY, batch_size=BS, shuffle=True)

# Build a Tensorflow dataset from the training data
dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

# Build the data input pipeline
print("[INFO] Creating a tf.data input pipeline...")
dataset = (dataset.shuffle(1024).cache().repeat().batch(BS).prefetch(AUTOTUNE))

# Benchmark the image data generator and display the number of data points generated,
# along with the time taken to perform the operation
total_time = benchmark(datasetGen=dataGen, numSteps=NUM_STEPS)
print("[INFO] ImageDataGenerator generated {} images in  {:.2f} seconds...".format(BS * NUM_STEPS, total_time))

# Create a dataset iterator, benchmark the tf.data.pipeline and display the number of data points generated,
# along with the time taken to perfrom the operation
datasetGen = iter(dataset)
total_time = benchmark(datasetGen, NUM_STEPS)
print("[INFO] tf.data generated {} images in {:.2f} seconds...".format(BS * NUM_STEPS, total_time))
