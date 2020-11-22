# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape


# -----------------------------
#   DCGAN CLASS
# -----------------------------
class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, inputDim=1, outputDim=512):
        # Initialize the model along with the input shape to be "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (dim, dim, depth)
        chanDim = -1
        # First set of FC =≳ RELU => BN layers
        model.add(Dense(input_dim=inputDim, units=outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # Second set of FC => RELU => BN layers, this time preparing the number of FC nodes to be reshaped into a volume
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        # Reshape the output of the previous layer set, upsample + apply a transposed convolution, RELU and BN
        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        # Apply another upsample and transposed convolution, but this time output the TANH activation
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding="same"))
        model.add(Activation("tanh"))
        # Return the generator model
        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        # Initialize the model along with the input shape to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        # First set of CONV => RELU layers
        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2), input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))
        # Second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))
        # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
        # Sigmoid layer outputting a single value
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        # Return the discriminator model
        return model
