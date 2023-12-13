"""
Convert YOLO segmentation labels (txt files) to image masks.

Functionalities:
- extract polygons from YOLO files
- filter category
- write mask image with options to either keep orig image in mask or generate binary image
- support multiple image formats

Author: Sebastian Haan
"""

import os
import numpy as np
import cv2 as cv
import shutil

def extract_masks(fname_img, fname_txt, cat_id = None):
    """
    extract image mask coordinates from txt file

    Args:
        fname_img (str): filename of the image
        fname_txt (str): filename of the txt file 
        cat_id (int): category id of label, default None (all labels converted to masks)                      
    
    Returns:
        pts: list of mask arrays
        img: image
    """
    ## Reading the image
    img = cv.imread(fname_img)
    height = img.shape[0]
    width = img.shape[1]

    with open(fname_txt) as txt:
        txt_lines = txt.readlines()
    pts=[]
    for line in txt_lines:
        try:
            line = line.split(" ")
            #line=list(filter(lambda i: "." in i,line))
            pts_part = []
            if cat_id != None and line[0] == str(cat_id):  
                for i in range(1, len(line), 2):
                    pts_part.append([int(float(line[i]) * width), int(float(line[i + 1]) * height)])
                pts.append(pts_part) 
        except:
            pass
    return pts, img 


def convert_to_mask(path_labels, path_images, outpath, cat_id = None, format_image= "png", format_out="jpg"):
    """
    Convert yolo labels to masks for a given category ID

    Args:
        path_labels (str): path to the folder with yolo labels
        path_images (str): path to the folder with corresponding images. Image names should be the same as label names.
        outpath (str): path to the folder where masks will be saved
        cat_id (int): category id of label, default None (all labels converted to masks)
        format_image (str): image format, default "png"         
        format_out (str): output image format, default "png"     
    """

    image_list = [f for f in os.listdir(path_images) if f.endswith(f'.{format_image.lower()}') or f.endswith(f'.{format_image.upper()}')]
    txt_list = [f for f in os.listdir(path_labels) if f.endswith(f'.txt')]

    for fname_txt in txt_list:
        # splitting the name of the txt to make filename of the image
        txt_num=fname_txt.split(".")[0] 

        # finding the corresponding image
        for fname_img in image_list:
            img_num=fname_img.split(".")[0]
            if img_num == txt_num:
                break

        pts, img = extract_masks(os.path.join(path_images,fname_img), os.path.join(path_labels,fname_txt), cat_id)

        ## making array of arrays
       # q = [np.asarray(n) for n in pts]
       # pts = np.array(q) 
        pts = [np.asarray(n) for n in pts]

        ## making the mask
        mask = np.zeros(img.shape[:2], np.uint8)
        #pts = np.array(pts, np.int32)
        # Draw the contour using the coordinate pairs and fill it with 255 (white) in the mask array:
        res = cv.drawContours(mask, pts, -1, (255, 255, 255), -1, cv.LINE_AA)

        ## applying the mask
        dst = cv.bitwise_and(img, img, mask=res)

        ## write masked image in a folder
        success = cv.imwrite(os.path.join(outpath,txt_num + f'mask_class{cat_id}.{format_out}'), dst)
        if not success:
            raise Exception(f"Could not write image {txt_num + f'mask_class{cat_id}.{format_out}'}")

 