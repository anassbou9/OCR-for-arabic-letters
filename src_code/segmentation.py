import numpy as np
import preprocessing as pre
import matplotlib.pyplot as plt
import os
import cv2 as cv
from itertools import chain


PROJECTION_DICT = {"vertical": 0, "horizontal": 1}
CURR_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_TEST_PATH = CURR_PATH + "/../books_for_ocr/scanned_pics/test_13.PNG"
line_path = CURR_PATH + "/../lines"
words_path = CURR_PATH + "/../words"



def projection(image, axis):

    ones_and_zeros = 1 - image // 255
    projection_bins = np.sum(ones_and_zeros, axis=PROJECTION_DICT[axis])
    return projection_bins

def gray_projection(image, axis):
    """ Compute the horizontal or the vertical projection of a gray image """

    if axis == 'horizontal':
        projection_bins = np.sum(image, 1).astype('int32')
    elif axis == 'vertical':
        projection_bins = np.sum(image, 0).astype('int32')

    return projection_bins


def segment(image, axis, thresh=17, cut=4):

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

def visualize(list_of_images, output_path):

    for i, line in enumerate(list_of_images):
        cv.imwrite(
            output_path + "/segment_" + str(i) + ".jpg",
            line,
        )


def extract_words(img, write = 0):

    lines = segment(img, "horizontal")
    words = []        

    for line in lines:
        line_words = segment(line, "vertical", 0, 6)
        line_words.reverse()
        for word in line_words:
            words.append((word,line))
    
    if write:

        visualize(lines, line_path)
        flatten_list = list(chain.from_iterable(words))
        visualize(flatten_list, words_path)

    return words





original_img, preprocessed_img = pre.preprocess(INPUT_TEST_PATH)
words = extract_words(preprocessed_img)

#lines = segment(preprocessed_img, "horizontal")


"""words = segment(lines[8], "vertical", 0, 6)
for i, word in enumerate(words):
    cv.imwrite(
        words_path + "/word)" + str(i) + ".jpg",
        word,
    )"""
