"""
Convert YOLO segmentation labels (txt files) to image masks.

Example usage via command line:
python yolo2mask.py --label_dir /path/to/labels --image_dir /path/to/images --output_dir /path/to/output --category_id 0 --format jpg

Required Arguments:
--label_dir: Directory of YOLO segmentation labels (txt files)
--image_dir: Directory of corresponding images to labels

Optional Arguments:
--output_dir: Output directory
--category_id: Selected category ID. This will filter based on the first number in each label row
--format: Output image format, default "jpg"
--keep_orig: Keep original image in the mask. Default False, binary mask will be generated


Functionalities:
- convert YOLO polygons to masks
- solve nested polygons by merging them
- filtering category
- write mask image with options to either keep orig image in mask or generate binary image
- support multiple image formats

Author: Sebastian Haan
"""

import os
import numpy as np
import cv2 as cv
import shutil
import argparse

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
    # Read image
    img = cv.imread(fname_img)
    height = img.shape[0]
    width = img.shape[1]

    with open(fname_txt) as txt:
        txt_lines = txt.readlines()
    pts=[]
    for line in txt_lines:
        try:
            line = line.split(" ")
            pts_part = []
            if cat_id is not None:
                if line[0] != str(cat_id): 
                    continue 
            for i in range(1, len(line), 2):
                pts_part.append([int(float(line[i]) * width), int(float(line[i + 1]) * height)])
            pts.append(pts_part) 
        except:
            pass
    return pts, img 


def convert_to_mask(path_labels, path_images, outpath = None, cat_id = None, format_out="jpg", keep_orig = False):
    """
    Convert yolo labels to masks for a given category ID.
    Generated masks are saved in outpath folder.

    Args:
        path_labels (str): path to the folder with yolo labels
        path_images (str): path to the folder with corresponding images. Image names should be the same as label names.
        outpath (str): path to the folder where masks will be saved. Default None, masks are saved in the same folder as labels.
        cat_id (int): category id of label, default None (all labels converted to masks)   
        format_out (str): output image format, default "png"     
        keep_orig (bool): if True, keep original image in the mask. Default False, binary mask is saved 
    """

    # check if output path is given
    if outpath is None:
        outpath = path_labels
    else:
        os.makedirs(outpath, exist_ok=True)

    # accepted input image formats
    accepted_formats = ["png", "jpg", "jpeg", "tif", "tiff"]
    image_list = [f for f in os.listdir(path_images) if f.split(".")[-1].lower() in accepted_formats]

    # label files
    txt_list = [f for f in os.listdir(path_labels) if f.endswith(f'.txt')]

    for fname_txt in txt_list:
        # splitting the name of the txt to make filename of the image
        txt_num=fname_txt.split(".")[0] 

        # finding the corresponding image
        for fname_img in image_list:
            img_num=fname_img.split(".")[0]
            if img_num == txt_num:
                break

        # extract polygons from YOLO txt file
        pts, img = extract_masks(os.path.join(path_images,fname_img), os.path.join(path_labels,fname_txt), cat_id)
        pts = [np.asarray(n) for n in pts]

        ## making the mask
        mask = np.zeros(img.shape[:2], np.uint8)
        # generate contour using pts and fill mask with 255 (white):
        for i in range(len(pts)):
            try:
                cv.drawContours(mask, pts, i, (255, 255, 255), thickness=-1, lineType=cv.LINE_AA)
            except:
                pass

        ## applying the mask
        if keep_orig:
            dst = cv.bitwise_and(img, img, mask=mask)
        else:
            dst = mask

        ## write masked image in a folder
        success = cv.imwrite(os.path.join(outpath,txt_num + f'.{format_out}'), dst)
        if not success:
            raise Exception(f"Could not write image {txt_num + f'.{format_out}'}")
        

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO segmentation labels (txt files) to image masks.')
    parser.add_argument('-l', '--label_dir', help='Directory of YOLO segmentation labels (txt files)', required=True)
    parser.add_argument('-i', '--image_dir', help='Directory of corresponding images to labels', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', default=None)
    parser.add_argument('-c', '--category_id', help='Selected category ID. This is the first number in each label row', default=None)
    parser.add_argument('-f', '--format', help='Output image format', default='jpg', choices=['png', 'jpg', 'tif'])
    parser.add_argument('-k', '--keep_orig', default = False, help='Keep original image in the mask. Default False, binary mask will be generated', action='store_true')
    args = parser.parse_args()

    convert_to_mask(args.label_dir, args.image_dir, args.output_dir, args.category_id, args.format, args.keep_orig)

if __name__ == '__main__':
    main()