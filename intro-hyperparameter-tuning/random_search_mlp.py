# -----------------------------
#   USAGE
# -----------------------------
# python random_search_mlp.py

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import tensorflow as tf
tf.random.set_seed(42)  # Import tensorflow and fix the random seed for better reproducibility
from pyimagesearch.mlp import get_mlp_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
print("[INFO] Downloading MNIST dataset...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# Scale data to the range [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# Wrap the model into a scikit-learn compatible classifier
print("[INFO] Initializing the model...")
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)

# Define a grid for the hyperparameter search space
hiddenLayerOne = [256, 512, 784]
hiddenLayerTwo = [128, 256, 512]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]

# Create a dictionary from the hyperparameter grid
grid = dict(hiddenLayerOne=hiddenLayerOne, learnRate=learnRate,
            hiddenLayerTwo=hiddenLayerTwo, dropout=dropout,
            batch_size=batchSize, epochs=epochs)

# Initialize a random search with a 3-fold cross-validation and then start the hyperparameter search process
print("[INFO] Performing random search...")
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3, param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(trainData, trainLabels)

# Summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] Best score is {:.2f} using {}".format(bestScore, bestParams))

# Extract the best model, make predictions on the data and show a classification report
print("[INFO] Evaluating the model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(testData, testLabels)
print("Accuracy: {:.2f}%".format(accuracy * 100))

