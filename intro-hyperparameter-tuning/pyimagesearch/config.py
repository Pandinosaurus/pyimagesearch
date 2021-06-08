# Specify the path to the dataset
CSV_PATH = "abalone_dataset/abalone_train.csv"

# Define the path to the output directory
OUTPUT_PATH = "output"

# Specify the column names of the dataframe
COLS = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Age"]

# Initialize the input shape and number of classes
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

# Define the total number of epochs to train, batch size and the early stopping patience
EPOCHS = 50
BS = 32
EARLY_STOPPING_PATIENCE = 5
