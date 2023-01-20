import numpy as np
import preprocessing as pre
import matplotlib.pyplot as plt
import os
import cv2 as cv

PROJECTION_DICT = {"vertical": 0, "horizontal": 1}


def projection(image, axis):

    ones_and_zeros = 1 - image // 255
    projection_bins = np.sum(ones_and_zeros, axis=PROJECTION_DICT[axis])
    return projection_bins


def segment(image, axis, thresh=14, cut=4):

    hist = projection(image, axis)
    cnt = 0
    segments = []
    start = -1
    for idx_2, my_bin in enumerate(hist):

        if my_bin > thresh:
            cnt = 0

        if my_bin > thresh and start == -1:
            start = idx_2

        if my_bin <= thresh and start != -1:

            cnt += 1
            if cnt >= cut:

                if axis == "horizontal":
                    segments.append(image[max(start - 1, 0) : idx_2][:])
                    start = -1
                    cnt = 0

                elif axis == "vertical":

                    segments.append(image[:, max(start - 1, 0) : idx_2])
                    start = -1
                    cnt = 0

    return segments


curr_direct_path = os.path.dirname(os.path.abspath(__file__))
input_test_path = curr_direct_path + "/../books_for_ocr/scanned_pics/test_2.PNG"
print(input_test_path)
original_img, preprocessed_img = pre.preprocess(input_test_path)
lines = segment(preprocessed_img, "horizontal")
line_path = curr_direct_path + "/../lines"
for i, line in enumerate(lines):
    cv.imwrite(
        line_path + "/line" + str(i) + ".jpg",
        line,
    )
words = segment(lines[0], "vertical", 0, 6)
words_path = curr_direct_path + "/../words"
for i, word in enumerate(words):
    cv.imwrite(
        words_path + "/word)" + str(i) + ".jpg",
        word,
    )
