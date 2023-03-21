from keras.models import load_model
import os
import char_seg
import cv2 as cv
import numpy as np
import joblib
import preprocessing as pre
import segmentation as my_segm

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_TEST_PATH = CURR_PATH + "/../books_for_ocr/scanned_pics/test_7.PNG"


def binarize(image):
    """This function binarizes  an image using the otsu algorithm
    and returns the original image and the binarized image"""
    image = image.astype(np.uint8)
    # Can we make the binarization better? --> for later
    # the 0 transforms autmatically the picture to grayscale

    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh


def preprocess_char(img):
    if img.shape[0] > 32 or img.shape[1] > 32:
        print("kant kber")
        img = binarize(img)

        img = cv.resize(img, (32, 32))
        img = np.expand_dims(img, axis=-1)
        img = img.reshape(32, 32).T
    else:
        img = binarize(img)
        output_shape = (32, 32)
        # Calculate the padding required on each side
        pad_height = output_shape[0] - img.shape[0]
        pad_width = output_shape[1] - img.shape[1]

        # Calculate the amount of padding required on each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image with zeros
        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            "constant",
            constant_values=0,
        )
        img = padded_img.reshape(32, 32).T

    return img


def predict_char(char_img):
    model = load_model(CURR_PATH + "/../models_and_lb/CNN_3.h5")
    lb = joblib.load(CURR_PATH + "/../models_and_lb/label_encoder.joblib")

    char = preprocess_char(char_img)
    img = np.array([char])
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction, axis=1)
    y_class_name = lb.inverse_transform(predicted_index)
    return y_class_name[0]


def predict_word(word_line):
    chars_list = char_seg.segment(word_line[1], word_line[0])
    predicted_word = ""
    for char in chars_list:
        predicted_char = predict_char(char)
        predicted_word += predicted_char

    return predicted_word


if __name__ == "__main__":
    original_img, preprocessed_img = pre.preprocess(INPUT_TEST_PATH)

    lines = my_segm.segment(preprocessed_img, "horizontal")
    line = lines[7]
    words = my_segm.segment(line, "vertical", 0, 6)
    word = words[1]
    word_line = (word, line)

    predicted_word = predict_word(word_line)
    print(predicted_word)
