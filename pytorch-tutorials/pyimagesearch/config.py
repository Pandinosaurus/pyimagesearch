# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import torch

# Specify the image dimension
IMAGE_SIZE = 224

# Specify the ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Determine the device that is going to be used for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Specify the path to the ImageNet labels
IN_LABELS = "data/labels/ilsvrc2012_wordnet_lemmas.txt"
