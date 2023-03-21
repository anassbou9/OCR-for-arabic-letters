import cv2
import numpy as np
import os


def hough_transform(image):
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # Apply Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    # Create output image
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    output[:, :, 0] = edges
    output[:, :, 1] = edges
    output[:, :, 2] = edges

    # Draw lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Return output image and lines
    return output, lines


import cv2
import numpy as np


def cca_line_segmentation(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform CCA to label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=4, ltype=cv2.CV_32S
    )
    print(stats)

    # Create output image
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    output[:, :, 0] = binary
    output[:, :, 1] = binary
    output[:, :, 2] = binary

    # Draw lines
    for i in range(1, num_labels):
        # Get the bounding box of the connected component
        x, y, w, h, area = stats[i]

        # Draw a red line around the bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Return output image
    return output


def bounding_box_segmentation(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Compute the bounding boxes around each contour
    boxes = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    # Sort the bounding boxes from top to bottom
    boxes = sorted(boxes, key=lambda x: x[1])

    # Draw the bounding boxes around each line of text
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the segmented lines
    cv2.imshow("Segmented Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load image
curr_direct_path = os.path.dirname(os.path.abspath(__file__))
input_test_path = curr_direct_path + "/../books_for_ocr/scanned_pics/test_13.PNG"
image = cv2.imread(input_test_path)

# Apply Hough transform and visualize results
# output, lines = hough_transform(image)
# output = cca_line_segmentation(image)
bounding_box_segmentation(image)

"""cv2.imshow('Original', image)
cv2.imshow('Hough Transform', output)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
