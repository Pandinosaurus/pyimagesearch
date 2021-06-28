# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import time


# -----------------------------
#   FUNCTIONS
# -----------------------------
def benchmark(datasetGen, numSteps):
    # Start the timer count
    start = time.time()
    # Loop over the provided number of steps
    for i in range(0, numSteps):
        # Get the next batch of data
        (images, labels) = next(datasetGen)
    # End the timer count
    end = time.time()
    # Return the difference between the end and start times
    return end - start

