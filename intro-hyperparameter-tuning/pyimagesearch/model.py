# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from . import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# -----------------------------
#   FUNCTIONS
# -----------------------------
def build_model(hp):
    # Initialize the model along with the input shape and channel dimension
    model = Sequential()
    inputShape = config.INPUT_SHAPE
    chanDim = -1
    # First CONV => RELU => POOL Layer set
    model.add(Conv2D(hp.Int("conv_1", min_value=32, max_value=96, step=32),
                     (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Second CONV => RELU => POOL Layer set
    model.add(Conv2D(hp.Int("conv_2", min_value=64, max_value=128, step=32), (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # First (and only) set of FC => RELU Layers
    model.add(Flatten())
    model.add(Dense(hp.Int("dense_units", min_value=256, max_value=768, step=256)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Softmax classifier
    model.add(Dense(config.NUM_CLASSES))
    model.add(Activation("softmax"))
    # Initialize the learning rate choices and optimizer
    lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])
    opt = Adam(learning_rate=lr)
    # Compile the model
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    # Return the model
    return model

