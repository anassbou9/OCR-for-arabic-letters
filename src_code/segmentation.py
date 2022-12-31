import numpy as np
import preprocessing as pre
import matplotlib.pyplot as plt
import cv2 as cv

PROJECTION_DICT = { "vertical": 0,"horizontal": 1}

def projection(image, axis):

    ones_and_zeros = 1 - image//255
    projection_bins = np.sum(ones_and_zeros,axis=PROJECTION_DICT[axis])
    return projection_bins


original_img, preprocessed_img = pre.preprocess("c:/Users/ab/Desktop/src code/books_for_ocr/scanned_pics/test_2.PNG")

hist = projection(preprocessed_img, "horizontal")
print("hist: ",hist)
def segment(image, hist, thresh=2):
    cnt=0
    cut=3
    lines = []
    start = -1
    for idx_2,my_bin in enumerate(hist):
        if my_bin > thresh and start==-1:
            start = idx_2

        if my_bin <= thresh and start!=-1:
            cnt += 1
            if cnt >= cut:
                lines.append(image[max(start-1,0):idx_2][:])
                """if len(lines) in [2,4,6,9]:
                    print("the starting index:{}, until:{}".format(start,idx_2) )
                    print("the hist:", hist[start:idx_2-1])"""
                start = -1
                cnt = 0
    return lines

segments = segment(preprocessed_img,hist,16)
titles = ['text','first line']
for i,seg in enumerate(segments):
    #print(seg)
    cv.imwrite("c:/Users/ab/Desktop/src code/lines/line)"+str(i)+".jpg",seg)
#pre.save_images("line",segments,"c:/Users/ab/Desktop/src code/lines/")
"""for i,segment in enumerate(segments):
    images = [preprocessed_img, segment]
    print(i)
    pre.plot_images(images, titles)"""
            


    
    







