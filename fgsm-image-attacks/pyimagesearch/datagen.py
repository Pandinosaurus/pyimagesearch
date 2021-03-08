# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from .fgsm import generate_image_adversary
import numpy as np


# -----------------------------
#   FUNCTIONS
# -----------------------------
def generate_adversarial_batch(model, total, images, labels, dims, eps=0.01):
    # Unpack the image dimensions into convenience variables
    (h, w, c) = dims
    # Construct a data generator here to loop indefinetly
    while True:
        # Initialize the perturbed images and labels
        perturbImages = []
        perturbLabels = []
        # Randomly sample indexes (without replacement) from the input data
        idxs = np.random.choice(range(0, len(images)), size=total, replace=False)
        # Loop over the indexes
        for i in idxs:
            # Grab the current image and label
            image = images[i]
            label = labels[i]
            # Generate an adversarial image
            adversary = generate_image_adversary(model, image.reshape(1, h, w, c), label, eps=eps)
            # Update the perturbed images and labels lists
            perturbImages.append(adversary.reshape(h, w, c))
            perturbLabels.append(label)
        # Yield the perturbed images and labels
        yield np.array(perturbImages), np.array(perturbLabels)
