# -----------------------------
#   USAGE
# -----------------------------
# python match_histograms.py --source empire_state_cloudy.png --reference empire_state_sunset.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Path to the input source image")
ap.add_argument("-r", "--reference", required=True, help="Path to the input reference image")
args = vars(ap.parse_args())

# Load the source and reference images
print("[INFO] Loading source and reference images...")
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

# Determine if we are performing multichannel histogram matching and then perform histogram matching itself
print("[INFO] Performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)

# Show the output images
cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey(0)

# Construct a figure to display the histogram plots for each channel before and after histogram matching was applied
(fig, axs) = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# Loop over the source image, reference image and output matched image
for (i, image) in enumerate((src, ref, matched)):
    # Convert the image from BGR to RGB channel ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Loop over the names of the channels in RGB order
    for (j, color) in enumerate(("red", "green", "blue")):
        # Compute a histogram for the current channel and plot it
        (hist, bins) = exposure.histogram(image[..., j], source_range="dtype")
        axs[j, i].plot(bins, hist/hist.max())
        # Compute the cumulative distribution function for the current channel and plot it
        (cdf, bins) = exposure.cumulative_distribution(image[..., j])
        axs[j, i].plot(bins, cdf)
        # Set the y-axis label of the current plot to be the name of the current color channel
        axs[j, 0].set_ylabel(color)

# Set the axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

# Display the output plots
plt.tight_layout()
plt.show()

