from copy import deepcopy, copy
from os import path, listdir

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import rotate

# After the object is rotated, noises appear - let's make up for them with a small error
from intelligent_placer_lib.image_processing import compress_image, find_polygon_and_objects

ERR = 30

def check_image(path_to_image):
    image_full_path = path_to_image
    ans = None
    if path.splitext(path_to_image)[1] == ".jpg":
        img = cv2.imread(image_full_path, cv2.COLOR_BGR2GRAY)
        img = compress_image(img, 60)
        pol, obj_set = find_polygon_and_objects(img)
        ans, _ = fit_in_polygon(pol, obj_set)

    return ans

""" Try to fit objects in polygon"""
def fit_in_polygon(polygon, objects: list):
    objects.sort(key=lambda x: x.area, reverse=True)
    # check area of objects and polygon
    common_area = 0
    for obj in objects:
        common_area += obj.area
    if common_area > polygon.area:
        return False, None

    # double the  area of polygon mask
    ex_polygon_mask = np.ones(np.asarray(polygon.mask.shape) * 2)
    polygon_mask_h, polygon_mask_w = polygon.mask.shape
    pos_y, pos_x = np.asarray(ex_polygon_mask.shape) // 2 - np.asarray(polygon.mask.shape) // 2
    ex_polygon_mask[pos_y:pos_y + polygon_mask_h, pos_x:pos_x + polygon_mask_w] = polygon.mask

    # try to fit
    for i, obj in enumerate(objects):
        y, x = obj.mask.shape
        #plt.imshow(ex_polygon_mask)
        #plt.show()
        #plt.imshow(obj.mask)
        #plt.show()
        if not try_to_fit_one_object(ex_polygon_mask, obj.mask, pos_y - y, pos_x - x):
            #if i == 0: # cannot fit first object => false
            return False, None
        elif len(objects) <= 1:
            return True, ex_polygon_mask
        #else:
        #    objects.pop(i)
    """else:
        print("recursive")
        new_order_list = copy(objects)
        new_order_list.pop(i)
        new_order_list.append(obj)
        fit_in_polygon(polygon, new_order_list)"""

    return True, ex_polygon_mask

""" Try to fit one object in polygon"""
def try_to_fit_one_object(extended_polygon_mask, object_mask, pos_x, pos_y):
    object_mask_height, object_mask_width = object_mask.shape
    delta = 15
    step_angle = 10
    max_angle = 360
    polygon_mask_height, polygon_mask_width = extended_polygon_mask.shape

    for y in range(pos_y, polygon_mask_height - object_mask_height, delta):
        for x in range(pos_x, polygon_mask_width - object_mask_width, delta):
            for angle in range(0, max_angle, step_angle):
                rotated_object_mask = rotate(object_mask, angle, reshape=True)
                rotated_object_mask_height, rotated_object_mask_width = rotated_object_mask.shape
                polygon_mask_cut = extended_polygon_mask[y:y + rotated_object_mask_height, x:x + rotated_object_mask_width]

                try:
                    overlay_areas = cv2.bitwise_and(polygon_mask_cut.astype(int), rotated_object_mask.astype(int))
                except:
                    continue
                if np.sum(overlay_areas) < ERR:
                    extended_polygon_mask[y:y + rotated_object_mask_height, x:x + rotated_object_mask_width] = \
                        cv2.bitwise_xor(polygon_mask_cut.astype(int), rotated_object_mask.astype(int)).astype(bool)

                    return True

    return False

