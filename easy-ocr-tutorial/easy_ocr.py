# ---------------------------------------------------------------
#   USAGE
# ---------------------------------------------------------------
# python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar

# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
from easyocr import Reader
import argparse
import cv2


# -----------------------------
#   FUNCTIONS
# -----------------------------
def cleanup_text(text):
    # Strip out non-ASCII text in order to draw the text on the image using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image to be OCR'd")
ap.add_argument("-l", "--langs", type=str, default="en", help="Comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1, help="Whether or not GPU should be used")
args = vars(ap.parse_args())

# Break the languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# Load the input image from disk
image = cv2.imread(args["image"])

# OCR the input image using EasyOCR
print("[INFO] OCR'ing the input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)

# Loop over the results
for (bbox, text, prob) in results:
    # Display the OCR'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))
    # Unpack the bounding box
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # Cleanup the text and draw the box surrounding the text along with the OCR'd text itself
    text = cleanup_text(text)
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

