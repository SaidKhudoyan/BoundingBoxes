#!/usr/bin/env python
# coding: utf-8

import os.path
import numpy as np
from skimage.measure import find_contours
import cv2 as cv
import os
import json

from PIL import Image
from tqdm import tqdm


#========================================================
# Define Parameters such as Path.
base_path = "/home/said/Schreibtisch/BoundingBox/data/"

path_img  = base_path + "01_images"
path_mask = base_path + "02_masks"
 
dir_json = base_path + "03_bbox"
dir_overlay = base_path + "04_overlay"

# Get images and corresponding masks
orig_imgs  = [os.path.join(path_img, file) for file in os.listdir(path_img)]
orig_masks = [os.path.join(path_mask, file) for file in os.listdir(path_mask)]

orig_imgs.sort()
orig_masks.sort()

#========================================================

def load_img(pict):
    """

    :param 
    ---------------------------------
    pict: Original color picture

    :return: 
    ---------------------------------
    np array of original image
    """
    img_org = Image.open(pict)
    img_org = np.array(img_org)
    return img_org

def load_mask(or_mask):
    """

    :param 
    ---------------------------------
    or_mask: Original color picture

    :return: 
    ---------------------------------
    np array of original mask
    """
    mask_img = Image.open(or_mask)
    mask = np.array(mask_img.convert('L'))
    return mask


def save_bb_json(dict_bbox: list, save_path, fname):
    """

    :param 
    ---------------------------------
    dict_bbox: Dictionary containing the bounding_box coordinates and a label
    save_path: The target directory, where the dict_bbox should be saved
    fname: fname=bbox (name of the created file)

    """

    if os.path.exists(save_path):
        with open(os.path.join(save_path, f"{fname}.json"), "w") as outfile:
            json.dump(dict_bbox, outfile)
    else:
        os.makedirs(save_path)
        with open(os.path.join(save_path, f"{fname}.json"), "w") as outfile:
            json.dump(dict_bbox, outfile)



def save_overlay_bbox(image, list_b_boxes, save_path, fname):
    """

    :param 
    ---------------------------------
    image: Original image
    list_b_boxes: List of all b_boxes related to the image
    fname: fname=bbox (name of the created file)
    save_path: Target folder to save results
    """

    overlay = np.copy(image)
    fname = fname + ".png"

    for box in list_b_boxes:
        # [Xmin, Xmax, Ymin, Ymax]
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        overlay = cv.rectangle(overlay, (int(box["wmin"]), int(box["hmin"])), (int(box["wmax"]), int(box["hmax"])), color, 3) 

        # Add text to each segment
        overlay = cv.putText(overlay, box["label"], (box["wmin"], box["hmin"]-5), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    
    # save plot
    path = os.path.abspath(dir_overlay)
    if os.path.exists(path):
        if not cv.imwrite(os.path.join(save_path , fname), overlay):
            raise ValueError("Imwrite did not work in If statement")

    else:
        os.makedirs(path)
        if not cv.imwrite(os.path.join(save_path , fname), overlay):
            raise ValueError("imwrite did not work in else statement")

# ========================================================


# Iterate over masks
for i in tqdm(range(len(orig_masks))):
    
    _, img_filename = os.path.split(orig_imgs[i])
    name = img_filename.split("_")
    name = "_".join(name[:-1]) + "_"
    
    img  = load_img(orig_imgs[i]) 
    mask = load_mask(orig_masks[i])
    
    _, gray_img = cv.threshold(mask, 127, 255, 0)

    # Use dilatation and erosion for quality purposes
    kernel = np.ones((3,3),np.uint8)
    dilated_image = cv.dilate(gray_img, kernel, iterations=2)
    erosion_image = cv.erode(dilated_image, kernel, iterations=2)

    # Get contour of data
    contours = find_contours(erosion_image, 0.8)
    dict_bboxes = {}
    bounding_boxes = []

    for contour in contours:
        dict_instance = {}
        hmin = int(np.min(contour[:, 0]))  # height
        hmax = int(np.max(contour[:, 0]))  # height
        wmin = int(np.min(contour[:, 1]))  # width
        wmax = int(np.max(contour[:, 1]))  # width


        bounding_boxes.append({"wmin": wmin, "hmin": hmin, "wmax": wmax, "hmax": hmax, "label": "empty"})
    dict_bboxes["bboxes"] = bounding_boxes  

    # delete first bbox for complete lc if there are sectors
    if len(dict_bboxes["bboxes"])>1:
        dict_bboxes["bboxes"].pop(0)

        
    # save json bbox
    save_bb_json(dict_bboxes, save_path=dir_json, fname=name+"bbox")
    
    # save image overlay        
    save_overlay_bbox(img, dict_bboxes["bboxes"], save_path=dir_overlay, fname=name+"overlay")


print("Finished process...")