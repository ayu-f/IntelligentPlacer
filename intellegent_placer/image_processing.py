import abc
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu, threshold_minimum, try_all_threshold
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import regionprops, label


class Object:
    def __init__(self, obj_image: np.ndarray, obj_name: str):
        self.name = obj_name
        self.orig_image = obj_image
        self.process_image = None
        self.contour_image = None
        self.set_chars()

    def set_chars(self):
        self.process_image, self.contour_image, self.properties = get_item_from_img(self.orig_image)
        self.mask = cv2.normalize(np.array(self.properties.image, dtype=np.int32), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


def get_object_mask(img: np.ndarray):
    labels = label(img)
    properties = regionprops(labels)

    center = (img.shape[0] / 2, img.shape[1] / 2)
    dist = np.array([pow(center[0] - p.centroid[0], 2) + pow(center[1] - p.centroid[1], 2) for p in properties])
    item = dist.argmin()
    mask = (labels == (item + 1))
    return mask, properties[item]


def compress_image(src: np.ndarray, scale_percent: int):
    new_size = (int(src.shape[1] * scale_percent / 100), int(src.shape[0] * scale_percent / 100))
    return cv2.resize(src, new_size)


def draw_contours_mask(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, [contours[0]], 0, (255, 0, 0), 2)
    return image


def get_item_from_img(img: np.ndarray):
    origin_image = np.copy(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_blur_gray = rgb2gray(gaussian(img, sigma=2.3, channel_axis=True))
    threshold_img = threshold_otsu(img_blur_gray)
    res_image = img_blur_gray <= threshold_img

    res_image = binary_closing(res_image, footprint=np.ones((15, 15)))
    res_image = binary_opening(res_image, footprint=np.ones((8, 8)))
    mask, properties = get_object_mask(res_image)

    cmask = (mask * 255).astype("uint8")
    res_image = cv2.bitwise_and(origin_image, origin_image, mask=cmask)
    return res_image, draw_contours_mask(cmask, origin_image), properties


def get_polygon_from_img(img: np.ndarray):
    img = rgb2gray(img)
    height, width = img.shape

    img = canny(gaussian(img, 1.6), sigma=1.6, low_threshold=0.1)
    img = img[70:height - 70, 70:width - 70]
    polygon_img = img[0:int(height), int(width / 3):width]
    return polygon_img
