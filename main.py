import copy
from copy import deepcopy

import matplotlib.pyplot as plt
import cv2
from intelligent_placer_lib.image_processing import ObjectBase, compress_image, read_images
from intelligent_placer_lib.image_processing import find_polygon_and_objects
from os import listdir, path
from intelligent_placer_lib.intelligent_placer import fit_in_polygon
from intelligent_placer_lib import intelligent_placer

def read_data():
    print(intellegent_placer.check_image("images/dataset/0.jpg"))
    """polygon = []
    object_set = []
    imagess = []
    img_path = "images/dataset/9.jpg"
    #for image_path in listdir(path_to_folder):
    #    image_full_path = path.join(path_to_folder, image_path)
    #    if path.splitext(image_path)[1] == ".jpg":
    image_full_path = img_path

    img = cv2.imread(image_full_path, cv2.COLOR_BGR2GRAY)
    img = compress_image(img, 60)
    pol, obj_set = find_polygon_and_objects(img)
    imagess.append(img)
    polygon.append(pol)
    object_set.append(obj_set)

    plt.imshow(imagess[0])
    plt.show()
    obb = copy.copy(obj_set[0])
    ans, _ = fit_in_polygon(polygon[0], object_set[0])
    print(ans)
    pass"""

def main():
    read_data()

main()
