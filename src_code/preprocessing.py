import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate 


def binarize(image):
    """This function binarizes  an image using the otsu algorithm
    and returns the original image and the binarized image"""

    #Can we make the binarization better? --> for later
    #the 0 transforms autmatically the picture to grayscale
    
    ret,thresh = cv.threshold(image,127,255,cv.THRESH_OTSU)
    #imgf = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    return  thresh


def find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist - np.mean(hist)) ** 2)
    return  score


def correct_skew(image):
    """this function corrects skewness using the profile projection method
    it takes as argument the image after binarization"""

    # the binary_image array only contains 0 and 1, to easily calculate the histogramn ktb elach l inversed
    inversed_binary_image = 1- image//255
    angles = np.arange(-3,3,0.1)
    scores = []
    for angle in angles:
        scores.append(find_score(inversed_binary_image, angle))
    best_score = np.max(np.array(scores))
    best_angle = angles[scores.index(best_score)]
    corrected_skew = rotate(image, best_angle, reshape=False, order=0, cval=255)
    
    return corrected_skew
    
    
def plot_images(images_list,images_names=["before","after"]):
    """this function plots two images"""

    plt.subplot(1,2,1),plt.imshow(images_list[0],'gray',vmin=0,vmax=255)
    plt.title(images_names[0])
    plt.xticks([]),plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(images_list[1],'gray',vmin=0,vmax=255)
    plt.title(images_names[1])
    plt.xticks([]),plt.yticks([])
    plt.show()

def save_images(name, images_list,path):

    for i,image in enumerate(images_list):
        cv.imwrite(path+name+str(i)+".jpg",image)

def remove_noise(image):
    #shoudl make it way better!
    return cv.fastNlMeansDenoising(image) 

def preprocess(image):
    """this function binarizes, corrects skew and remove noise from the given
    picture"""

    img = cv.imread(image,0)
    binarized = binarize(img)
    corrected_skew = correct_skew(binarized)
    #images = [binarized, corrected_skew]
    #plot_images(images)
    removed_noise = remove_noise(corrected_skew)
    cv.imwrite("c:/Users/ab/Desktop/src code/lines/binarized.jpg",removed_noise)

    return img, removed_noise

#og, final = preprocess("skewed_2.png")

"""def thin_image(image):"""
"""ths function takes a bnary image as argument and returns
the image after performing thining and skeletonization""""""

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(image,kernel,iterations = 1)
return erosion"""



