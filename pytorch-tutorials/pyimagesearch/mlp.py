# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from collections import OrderedDict
import torch.nn as nn


# -----------------------------
#   FUNCTIONS
# -----------------------------
def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
    # Construct a shallow, sequential neural network
    mlpModel = nn.Sequential(OrderedDict([("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
                                          ("activation_1", nn.ReLU()),
                                          ("output_layer", nn.Linear(hiddenDim, nbClasses))]))
    # Return the sequential model
    return mlpModel
