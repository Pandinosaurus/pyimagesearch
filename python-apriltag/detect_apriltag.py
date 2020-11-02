# -----------------------------
#   USAGE
# -----------------------------
# python detect_apriltag.py --image images/example_01.png

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import apriltag
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image containing the AprilTag")
args = vars(ap.parse_args())

# Load the input image and convert the image to grayscale
print("[INFO] Loading the image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the AprilTag detector options and then detect the AprilTags in the input image
print("[INFO] Detecting AprilTags...")
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} Total AprilTags Detected!".format(len(results)))

# Loop over the AprilTags detection results
for r in results:
    # Extract the bounding box (x, y) coordinates for the AprilTag and convert each one of the (x,y) coordinate
    # pairs to integers
    (ptA, ptB, ptC, ptD) = r.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))
    # Draw the bounding box of the AprilTag direction using the extracted (x, y) coordinates
    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
    # Draw the center (x, y) coordinates of the AprilTag
    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
    # Draw the tag family on the image
    tagFamily = r.tag_family.decode("utf-8")
    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("[INFO] Tag Family: {}".format(tagFamily))

# Show the output image after the AprilTag detections
cv2.imshow("Image", image)
cv2.waitKey(0)


