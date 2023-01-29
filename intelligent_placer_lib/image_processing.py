
from os import listdir, path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, binary_opening
from skimage.measure import regionprops, label


RATIO = 0.7

""" Class for primitives and polygon"""
class ObjectBase:
    def __init__(self, obj_image: np.ndarray, obj_name: str, mask=None, global_mask=None):
        self.name = obj_name
        self.orig_image = obj_image
        self.process_image = None
        self.contour_image = None
        self.points, self.desc = None, None
        self.mask = mask
        self.area = None
        self.global_mask = global_mask
        if mask is None:
            self.set_chars()
        self.points, self.desc = find_points(self.orig_image)

    def set_chars(self):
        self.process_image, self.contour_image, self.properties = get_object_from_img(self.orig_image)
        self.mask = cv2.normalize(np.array(self.properties.image, dtype=np.int32), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    def setArea(self, mask_area):
        self.area = mask_area

    """ Match object"""
    def match(self, target) -> float:
        if len(self.points) < len(target.points):
            return target.match(self)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(target.desc, self.desc, k=2)

        choose = 0
        for m, n in matches:
            if m.distance > n.distance * RATIO:
                choose += 1
        return choose / len(target.points)

""" Get mask by image of primitives in paper"""
def get_object_mask(img: np.ndarray):
    labels = label(img)
    properties = regionprops(labels)

    center = (img.shape[0] / 2, img.shape[1] / 2)
    dist = np.array([pow(center[0] - p.centroid[0], 2) + pow(center[1] - p.centroid[1], 2) for p in properties])
    item = dist.argmin()
    mask = (labels == (item + 1))
    return mask, properties[item]

""" Find features """
def find_points(img: np.ndarray):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(img, None)

""" Compress image for more fast work"""
def compress_image(src: np.ndarray, scale_percent: int):
    new_size = (int(src.shape[1] * scale_percent / 100), int(src.shape[0] * scale_percent / 100))
    return cv2.resize(src, new_size)


def draw_contours_mask(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, [contours[0]], 0, (255, 0, 0), 2)
    return image

""" Read images and create ObjectBase's"""
def read_images(path_to_folder):
    for image_path in listdir(path_to_folder):
        image_full_path = path.join(path_to_folder, image_path)
        if path.splitext(image_path)[1] == ".jpg":
            img = cv2.imread(image_full_path)
            img = compress_image(img, 60)
            result = ObjectBase(img, image_full_path)

""" Get object and properties """
def get_object_from_img(img: np.ndarray):
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

""" Create mask from contour"""
def get_mask_from_contour(contours, fill=False, err=0) -> []:
    masks = []
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        x_most_left, width, y_most_bottom, height = bbox[1], bbox[3], bbox[0], bbox[2]
        x, y, w, h = cv2.boundingRect(cnt)

        mask = np.full((width, height), fill, dtype=bool)
        for y in range(y_most_bottom, y_most_bottom + height):
            for x in range(x_most_left, x_most_left + width):
                if cv2.pointPolygonTest(cnt, (y, x), True) >= err:
                    mask[x - x_most_left][y - y_most_bottom] = not fill
        masks.append(mask)

    return masks

""" Get polygon and objects in img"""
def find_polygon_and_objects(img):
    read_images("images/objects")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # process image for finding contours
    img_gauss = cv2.GaussianBlur(img_gray, (3,3), 0)
    img_canny = cv2.Canny(img_gauss, 80, 250)

    se = np.ones((5, 5), dtype='uint8')
    image_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, se)
    img_canny = image_close
    contours, hierarchy = cv2.findContours(image=img_canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    good_contours = []
    cx = []
    cy = []
    area_contours = []
    min_contour_area = 80

    # delete noise contours
    for i, cnt in enumerate(contours):
        contour_area = cv2.contourArea(cnt)
        img_copy = img.copy()
        cv2.drawContours(img_copy, [cnt], 0, (255, 0, 0), 8)
        if contour_area > min_contour_area and hierarchy[0][i][3] == -1:
            area_contours.append(contour_area)
            good_contours.append(cnt)
            M = cv2.moments(cnt)
            cx.append(int(M['m10'] / (M['m00'] + 1e-5)))
            cy.append(int(M['m01'] / (M['m00'] + 1e-5)))
    selected_contour = None
    x_max = 0
    idx = 0

    # filter contours
    for i, cnt in enumerate(area_contours):
        for j, cnt in enumerate(area_contours):
            if i != j and abs(area_contours[i] - area_contours[j]) <= 50 and abs(cx[i] - cx[j]) < 10 and abs(cy[i] - cy[j]):
                area_contours.pop(j)
                good_contours.pop(j)

    # detect polygon
    for i, cnt in enumerate(good_contours):
        x_pos = max([np.ndarray.reshape(x, (2,))[0] for x in cnt])
        img_copy = img.copy()
        cv2.drawContours(img_copy, [cnt], 0, (255, 0, 0), 8)
        if x_pos > x_max:
            selected_contour = cnt
            x_max = x_pos
            idx = i

    good_contours.pop(idx)
    img_copy = img.copy()
    # detect objects
    cv2.drawContours(img_copy, good_contours, -1, (255, 0, 0), 8)
    polygon = ObjectBase(img, "polygon", ~get_mask_from_contour([selected_contour], err=-1)[0])
    polygon.setArea(area_contours[idx])
    area_contours.pop(idx)
    labels = label(polygon.mask)
    properties = regionprops(labels)
    objects_set = []
    for i, mask in enumerate(get_mask_from_contour(good_contours)):
        objects_set.append(ObjectBase(img, f"{i}", (mask * 255).astype(int)))
        objects_set[i].setArea(area_contours[i])

    return polygon, objects_set
