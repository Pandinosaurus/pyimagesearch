# -----------------------------
#   USAGE
# -----------------------------
# python train.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Load the contents of the CVS annotations file
print("[INFO] Loading dataset...")
rows = open(config.ANNOTATIONS_PATH).read().strip().split("\n")

# Initialize the list of data (images), the target output predictions (bounding box coordinates),
# along with the filenames of the individual images
data = []
targets = []
filenames = []

# Loop over the rows
for row in rows:
    # Break the row into the filename and bounding box coordinates
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row
    # Derive the path to the input image, load the image (in OpenCV format), and grab its dimensions
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    # Scale the bounding box coordinates relative to the spatial dimensions of the input image
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    # Load the image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    # Update the list of data, targets and filenames
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

# Convert the data and targets to NumPy arrays, scaling the input pixel intensities, from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# Partition the data into training and testing splits using 90% of the data for training and 10% of the data for testing
split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

# Write the testing filenames to disk in order to use them when evaluating/testing the bounding box regression model
print("[INFO] Saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

# Load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze all VGG layers so they will *not* be updated during the training process
vgg.trainable = False

# Flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# Construct a fully-connected layer header to output the predicted bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# Construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# Initialize the optimizer, compile the model, and show the model summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# Train the network for bounding box regression
print("[INFO] Training the bounding box regression model...")
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),
              batch_size=config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=1)

# Serialize the model to disk
print("[INFO] Saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# Plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)