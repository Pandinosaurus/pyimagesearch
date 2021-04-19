# -----------------------------------
#   FUNCTIONS
# -----------------------------------
def convert_and_trim_bb(image, rect):
    # Extract the starting and ending (x,y) coordinates of the bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # Ensure that the bounding box coordinates fall within the spatial dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # Compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # Return the bounding box coordinates
    return startX, startY, w, h

