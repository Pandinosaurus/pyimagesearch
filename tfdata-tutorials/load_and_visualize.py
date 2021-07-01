# -----------------------------
#   USAGE
# -----------------------------
# python load_and_visualize.py --dataset datasets/animals
# python load_and_visualize.py --dataset datasets/animals --aug 1 --type layers
# python load_and_visualize.py --dataset datasets/animals --aug 1 --type ops


# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os


# -----------------------------
#   FUNCTIONS
# -----------------------------
def load_images(imagePath):
    # Read the image from disk, decode it, convert the data type to floating point and then resize it
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (156, 156))
    # Parse the class label from the file path
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    # Return the image and the label
    return image, label


def augment_using_layers(images, labels, aug):
    # Pass a batch of images through the data augmentation pipeline and return the augmented images
    images = aug(images)
    # Return the image and the label
    return images, labels


def augment_using_ops(images, labels):
    # Randomly flip the images horizontally and vertically, and rotate the images by 90 degrees in the counter
    # clock-wise direction
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.rot90(images)
    # Return the image and the label
    return images, labels


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input images dataset")
ap.add_argument("-a", "--augment", type=bool, default=False,
                help="Flag indicating whether or not augmentation will be applied")
ap.add_argument("-t", "--type", choices=["layers", "ops"], help="Method to be used to perform data augmentation")
args = vars(ap.parse_args())

# Set the batch size
BATCH_SIZE = 8

# Grab all image paths
imagePaths = list(paths.list_images(args["dataset"]))

# Build the dataset and data input pipeline
print("[INFO] Loading the dataset...")
ds = tf.data.Dataset.from_tensor_slices(imagePaths)
ds = (ds
      .shuffle(len(imagePaths), seed=42)
      .map(load_images, num_parallel_calls=AUTOTUNE)
      .cache()
      .batch(BATCH_SIZE)
      )

# Check the data augmentation flag
if args["augment"]:
    # Check if it is data augmentation with layers
    if args["type"] == "layers":
        # Initialize the sequential data augmentation pipeline
        aug = tf.keras.Sequential([preprocessing.RandomFlip("horizontal_and_vertical"),
                                   preprocessing.RandomZoom(height_factor=(-0.05, -0.15), width_factor=(-0.05, -0.15)),
                                   preprocessing.RandomRotation(0.3)])
        # Add data augmentation to the data input pipeline
        ds = (ds.map(lambda x, y: augment_using_layers(x, y, aug), num_parallel_calls=AUTOTUNE))
    # Otherwise, use data augmentation with Tensorflow image operations
    else:
        # Add data augmentation to the data input pipeline
        ds = (ds.map(augment_using_ops, num_parallel_calls=AUTOTUNE))

# Complete the data input pipeline
ds = (ds.prefetch(AUTOTUNE))

# Grab a batch of data from the dataset
batch = next(iter(ds))

# Initialize a figure
print("[INFO] Visualizing the first batch of the dataset...")
title = "With data augmentation {}".format("applied ({})".format(args["type"]) if args["augment"] else "not applied")
fig = plt.figure(figsize=(BATCH_SIZE, BATCH_SIZE))
fig.suptitle(title)

# Loop over the batch size
for i in range(0, BATCH_SIZE):
    # Grab the image and label from the batch
    (image, label) = (batch[0][i], batch[1][i])
    # Create a subplot and plot the image and label
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(image.numpy())
    plt.title(label.numpy().decode("UTF-8"))
    plt.axis("off")
    
# Show the plot
plt.tight_layout()
plt.show()

