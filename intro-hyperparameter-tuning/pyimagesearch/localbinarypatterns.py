# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from skimage import feature
import numpy as np


# -----------------------------
#  LOCAL BINARY PATTERNS CLASS
# -----------------------------
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # Store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # Compute the local binary pattern representation of the image,
        # and then use the LBP representation to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # Return the histogram of Local Binary Patterns
        return hist
