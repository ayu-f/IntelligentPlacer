# This is a sample Python script.
import imghdr
import intellegent_placer.image_processing
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import binary_closing
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from intellegent_placer.image_processing import ObjectBase, compress_image, read_images
from intellegent_placer.image_processing import find_polygon_and_objects
from os import listdir, path


def read_image(path_to_folder):
    for image_path in listdir(path_to_folder):
        image_full_path = path.join(path_to_folder, image_path)
        if path.splitext(image_path)[1] == ".jpg":
            img = cv2.imread(image_full_path)
            img = compress_image(img, 60)
            result = ObjectBase(img, image_full_path)
            processed_items = [(result.mask * 255).astype("uint8"), result.contour_image]
            #result = get_polygon_from_img(img)
            #cv2.imwrite("tmp.jpg", result.contour_image)
            plt.imshow(result.contour_image)
            plt.show()
    pass

def read_data(path_to_folder):
    polygon = []
    object_set = []
    for image_path in listdir(path_to_folder):
        image_full_path = path.join(path_to_folder, image_path)
        if path.splitext(image_path)[1] == ".jpg":
            img = cv2.imread(image_full_path, cv2.COLOR_BGR2GRAY)
            img = compress_image(img, 60)
            pol, obj_set = find_polygon_and_objects(img)
            polygon.append(pol)
            object_set.append(obj_set)
            #plt.imshow(result)
            #plt.show()
    for obj in object_set[6]:
        plt.imshow(obj.mask)
        plt.show()
    pass

def main():

    path_to_data = "images/dataset"
    #image_path = "3_1.jpg"
    #read_image(path_to_obj)
    read_data(path_to_data)
    """for image_path in listdir(path_to_folder):
        image_full_path = path.join(path_to_folder, image_path)
        if path.splitext(image_path)[1] == ".jpg":
            img = cv2.imread(image_full_path)
            img = compress_image(img, 60)
            result = Object(img, image_full_path)
            processed_items = [(result.mask * 255).astype("uint8"), result.contour_image]
            #result = get_polygon_from_img(img)
            #cv2.imwrite("tmp.jpg", result.contour_image)
            plt.imshow(result.contour_image)
            plt.show()
    pass"""


main()
