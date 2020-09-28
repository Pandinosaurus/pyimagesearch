# ------------------------
#   USAGE
# ------------------------
# python mask_rcnn_grabcut.py --mask-rcnn mask-rcnn-coco --image example.jpg

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True, help="base path to mask-rcnn directory")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0, help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-e", "--iter", type=int, default=10, help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())

# Load the COCO class labels that the Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"], "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"], "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"], "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# Load the Mask R-CNN trained on the COCO dataset (90 classes) from disk
print("[INFO] Loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# Check use of GPU
if args["use_gpu"]:
    # Set CUDA as the preferable backend and target
    print("[INFO] Setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load the input image from disk and display it to the screen
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
cv2.imshow("Input", image)

# Construct a blob from the input image and then perform a forward pass of the Mask R-CNN giving us:
# (1) The bounding box coordinates of the objects
# (2) The pixel-wise segmentation for each specific object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

# Loop over the number of detected objects
for i in range(0, boxes.shape[2]):
    # Extract the class ID of the detection along with the confidence (i.e, probability) associated with the prediction
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]
    # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
    if confidence > args["confidence"]:
        # Show the class label
        print("[INFO] Showing output for '{}'...".format(LABELS[classID]))
        # Scale the bounding box coordinates back relative to the size of the image and then compute the width
        # and the height of the bounding box
        (H, W) = image.shape[:2]
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        boxW = endX - startX
        boxH = endY - startY
        # Extract the pixel-wise segmentation for the object, resize the mask such that it's the same dimensions as the
        # bounding box and then finally threshold to create a *binary* mask
        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
        mask = (mask > args["threshold"]).astype("uint8") * 255
        # Allocate memory for the output Mask R-CNN mask and store the predicted Mask R-CNN mask in the GrabCut mask
        rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
        rcnnMask[startY:endY, startX:endX] = mask
        # Apply a bitwise AND to the input image to show the output of applying the Mask R-CNN mask to the image
        rcnnOutput = cv2.bitwise_and(image, image, mask=rcnnMask)
        # Show the output of the Mask R-CNN and bitwise AND operation
        cv2.imshow("R-CNN Mask", rcnnMask)
        cv2.imshow("R-CNN Output", rcnnOutput)
        cv2.waitKey(0)
        # Clone the Mask R-CNN mask in order to use it when applying the GrabCut and then set any mask values greater
        # than zero to be "probable foreground" (otherwise these values are "definite background")
        gcMask = rcnnMask.copy()
        gcMask[gcMask > 0] = cv2.GC_PR_FGD
        gcMask[gcMask == 0] = cv2.GC_BGD
        # Allocate memory for two arrays that the GrabCut algorithm internally uses when segmenting the foreground
        # from the background and then apply the GrabCut using the mask segmentation method
        print("[INFO] Applying GrabCut to '{}' ROI...".format(LABELS[classID]))
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        (gcMask, bgModel, fgModel) = cv2.grabCut(image, gcMask, None, bgModel, fgModel, iterCount=args["iter"],
                                                 mode=cv2.GC_INIT_WITH_MASK)
        # Set all definite background and probable background pixels to 0 while definite foreground
        # and probable foreground pixels are set to 1, then scale the mask from the range [0, 1] to [0, 255]
        outputMask = np.where((gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
        outputMask = (outputMask * 255).astype("uint8")
        # Apply a bitwise AND to the image using the mask generated by GrabCut to generate the final output image
        output = cv2.bitwise_and(image, image, mask=outputMask)
        # Show the output GrabCut mask as well as the output of applying the GrabCut mask to the original input image
        cv2.imshow("GrabCut Mask", outputMask)
        cv2.imshow("Output", output)
        cv2.waitKey(0)

