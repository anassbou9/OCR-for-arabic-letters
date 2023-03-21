import cv2 as cv
import preprocessing as pre
import segmentation as my_segm
import predict_word as pred_w
import os 
import post_processing


CURR_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_TEST_PATH = CURR_PATH + "/../books_for_ocr/scanned_pics/test_16.png"

def run_ocr(img_path):


    original_img, preprocessed_img = pre.preprocess(img_path)
    predicted_text = ""
    words = my_segm.extract_words(preprocessed_img)
    predicted_words = list(map(pred_w.predict_word, words))

    for word in predicted_words:
        predicted_text += word
        predicted_text += " "
    
    return predicted_text

if __name__ == "__main__":

    text = run_ocr(INPUT_TEST_PATH)
    post_processing.post_process(text)
    print(text)
