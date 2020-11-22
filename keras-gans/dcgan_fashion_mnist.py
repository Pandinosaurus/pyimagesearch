# -----------------------------
#   USAGE
# -----------------------------
# python dcgan_fashion_mnist.py --output output

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from pyimagesearch.dcgan import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50, help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128, help="batch size for training")
args = vars(ap.parse_args())

# Store the epochs and batch size in convenience variables, then initialize the learning rate
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4

# Load the Fashion MNIST dataset and stack the training and testing data point in order to have additional training data
print("[INFO] Loading MNIST dataset...")
((trainX, _), (testX, _)) = fashion_mnist.load_data()
trainImages = np.concatenate([trainX, testX])

# Add in an extra dimension for the channel and scale the images into the range [-1,1]
# (which is the range of tanh function)
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5

# Build the generator
print("[INFO] Building the generator...")
gen = DCGAN.build_generator(7, 64, channels=1)

# Build the discriminator
print("[INFO] Building discriminator...")
disc = DCGAN.build_discriminator(28, 28, 1)
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)

# Build the adversarial model by first setting the discriminator to *not* be trainable,
# then combine the generator and discriminator together
print("[INFO] Building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

# Compile the GAN
ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)

# Randomly generate some benchmark noise so we can consistently visualize how the generative modeling is learning
print("[INFO] Starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

# Loop over the epochs
for epoch in range(0, NUM_EPOCHS):
    # Show epoch information and compute the number of batches per epoch
    print("[INFO] Starting epoch {} of {}...".format(epoch + 1, NUM_EPOCHS))
    batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)
    # Loop over the batches
    for i in range(0, batchesPerEpoch):
        # Initialize an (empty) output path
        p = None
        # Select the next batch of images, then randomly generate noise for the generator to predict on
        imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        # Generate images using the noise + generator model
        genImages = gen.predict(noise, verbose=0)
        # Concatenate the *actual* images and the *generated* images, construct class labels for the discriminator,
        # and shuffle the data
        X = np.concatenate((imageBatch, genImages))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
        y = np.reshape(y, (-1,))
        (X, y) = shuffle(X, y)
        # Train the discriminator on the data
        discLoss = disc.train_on_batch(X, y)
        # Let's now train our generator via the adversarial model by:
        # (1) generating random noise
        # (2) training the generator with the discriminator weights frozen
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        fakeLabels = [1] * BATCH_SIZE
        fakeLabels = np.reshape(fakeLabels, (-1,))
        ganLoss = gan.train_on_batch(noise, fakeLabels)
        # Check to see if this is the end of an epoch, and if so, initialize the output path
        if i == batchesPerEpoch - 1:
            p = [args["output"], "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]
        # Otherwise, check to see if we should visualize the current batch for the epoch
        else:
            # Create more visualizations early in the training process
            if epoch < 10 and i % 25 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]
            # Visualizations later in the training process are less interesting
            elif epoch >= 10 and i % 100 == 0:
                p = [args["output"], "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]
        # Check to visualize the output of the generator model on our benchmark data
        if p is not None:
            # Show loss information
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, " 
                  "adversarial_loss={:.6f}".format(epoch + 1, i, discLoss, ganLoss))
            # Make predictions on the benchmark noise, scale it back to the range [0, 255], and generate the montage
            images = gen.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]
            # Write the visualization to disk
            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)