# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf


# -----------------------------
#   FUNCTIONS
# -----------------------------
def contrastive_loss(y, preds, margin=1):
    # Explicitly cast the true class label data type to the predicted class label data type
    # (Otherwise we run the risk of having two separate data types, causing Tensorflow to error out)
    y = tf.cast(y, preds.dtype)
    # Calculate the contrastive loss between the true labels and the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # Return the computed contrastive loss to the calling function
    return loss

