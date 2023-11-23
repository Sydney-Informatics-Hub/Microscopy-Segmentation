"""
Convert segmentation annotations in raster format to COCO json format.
Raster annotations are images (stack of images) that encode segmentation masks in grayscale. 
By default, this program assumes only one object category and polygons are extracted from each grayscale segment that has a value larger than 0.
Optionally, you can set the argument "extract_segment_grayscale", which will use the grayscale value as a category ID.

Example usage:
python raster2coco.py --path_annotation /path/to/annotation/files --path_img /path/to/image/files --format_img tif --outfname_coco /path/to/output.json

Required Args:
    path_annotation (str): Path to raster annotation files.
    path_img (str): Path to image files. Used to extract metadata needed and to add information to json.

Optional Args:
    format_img (str): Format of image files. Default: 'tif'.
    outfname_coco (str): Path to output json file. If None, a json file with the same name as the model file will be created.
    extract_segment_grayscale: If True, the grayscale value inside the polygon is calculated and added to the dataframe as objectID. Default: False.

Author: Sebastian Haan
"""

import cv2
import numpy as np
import os
import re
import argparse
import datetime
import pandas as pd
import subprocess
from PIL import Image
import json
from detectron2.structures import BoxMode

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
        
def get_image_size(image_path):
    """Get the dimensions of an image.

    Args:
        image_path (str): The path to the image.

    Returns:
        tuple: The width and height of the image in pixels.
    """
    with Image.open(image_path) as img:
        return img.size

def extract_numeric_characters(string):
    """
    Extracts numeric characters from string.
    """
    numeric_chars = re.findall(r'\d+', string)
    return ''.join(numeric_chars)

def generate_coco_info(description, version = None, contributor = None):
    """
    Generate COCO info dictionary.

    Parameters
    ----------
    description : str
        Description of the dataset.
    version : str
        Version of the dataset. Default: None.
    contributor : str
        Contributor of the dataset. Default: None.

    Returns
    -------
    info : dict
    """
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if version is None:
        version = "none"
    if contributor is None:
        contributor = "none"
    info = {
        "description": description,
        "version": version,
        "year": datetime.datetime.now().year,
        "contributor": contributor,
        "date_created": current_date
    }
    return info

def get_bbox_from_points(points):
    """
    Get bounding box from list of points.

    Parameters
    ----------
    points : list
        List of points in format [[x1,y1], [x2,y2], ...]

    Returns
    -------
    bbox : list
        Bounding box in format [x, y, width, height]
    """
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    bbox = [min(x), min(y), max(x)-min(x), max(y)-min(y)]
    return bbox

def validate_coco_json(file_path):
    """
    Validates COCO json file against COCO json schema.

    Example usage:
    validate_coco_json('path_to_your_file.json')

    Parameters
    ----------
    file_path : json file
        Json file to be validated.

    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    assert isinstance(data, dict), "Root must be a dictionary."
    
    for key in ["info", "images", "annotations", "categories"]:
        assert key in data, f"'{key}' not in root dictionary."
    
    assert isinstance(data["info"], dict), "'info' must be a dictionary."
    for key in ["description", "version", "year", "contributor", "date_created"]:
        assert key in data["info"], f"'{key}' not in 'info'."
    
    for key in ["images", "annotations", "categories"]:
        assert isinstance(data[key], list), f"'{key}' must be a list."
    
    for image in data["images"]:
        assert isinstance(image, dict), "Each element in 'images' must be a dictionary."
        for key in ["file_name", "height", "width", "id"]:
            assert key in image, f"'{key}' not in an image."
    
    for annotation in data["annotations"]:
        assert isinstance(annotation, dict), "Each element in 'annotations' must be a dictionary."
        for key in ["id", "image_id", "category_id", "segmentation"]:
            assert key in annotation, f"'{key}' not in an annotation."
    
    for category in data["categories"]:
        assert isinstance(category, dict), "Each element in 'categories' must be a dictionary."
        for key in ["supercategory", "id", "name"]:
            assert key in category, f"'{key}' not in a category."


def generate_coco_dict(
    df, 
    imgfile, 
    coco_dict = None, 
    imgheight = None, 
    imgwidth = None, 
    imgformat = 'tif', 
    coco_info = None, 
    licenses = None,
    calc_bbox = True):
    """
    Generate COCO dict from dataframe.
    Here the dict format follows the format of the COCO dataset.

    Parameters
    ----------
    df : pandas dataframe
        must include at least the following columns: ['objectId', polygon]
    imgfile : str
        Path to image file.
    coco_dict : json file
        COCO json file that will be appeded with the new image and annotations.
        If None, a new json file will be created.
    imgheight : int
        Height of image in pixels. If None, extracted from image file.
    imgwidth : int
        Width of image in pixels. If None, extracted from image file.
    imgformat : str
        Format of image file. Default: 'tif'.
    coco_info : dict
        Dictionary with information about the dataset. Default: None.
    licenses : dict
        Dictionary with information about the licenses. Default: None.
    calc_bbox : bool
        If True, bounding box will be calculated from points. Default: True.
    
    Returns
    -------
    coco_dict : dict
    """
    # if imgheight or imgwidth are not provided, get them from image file
    if (imgheight is None) or (imgwidth is None):
        imgwidth, imgheight = get_image_size(imgfile)

    # get name of image file without path
    imgfile_name = os.path.basename(imgfile)

    # create json dictionary
    if coco_dict is None:
        #if not isinstance(licenses, list):
        #    licenses = [licenses]
        if coco_info is None:
            coco_info = generate_coco_info(description = 'IMOD model converted to COCO format')
        coco_dict = {
            "info": coco_info,
            "licenses": licenses,
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        img_id = 1
        ann_id = 1
    else:
        img_id = coco_dict['images'][-1]['id'] + 1
        ann_id = coco_dict['annotations'][-1]['id'] + 1

    coco_dict['images'].append({
        "file_name": imgfile_name,
        "height": imgheight,
        "width": imgwidth,
        "id": img_id
    })

    # get unique objectIds
    objectIds = df['objectId'].unique()
    # loop over all objectIds
    for objectId in objectIds:
        # add category if not already in categories
        if not any(d.get('id', None) == objectId for d in coco_dict['categories']):
            coco_dict['categories'].append({
                "supercategory": "none",
                "id": objectId,
                "name": "class_" + str(objectId)
                })
        # get all points for this objectId
        df_obj = df[df['objectId'] == objectId].copy()
        # loop over all contourIds
        contourIds = df_obj['contourId'].unique()
        for contourId in contourIds:
            # get all points for this contourId
            points = df_obj[df_obj['contourId'] == contourId]['polygon'].values[0].tolist()
            # reverse y coordinates with respect to image height
            #points = [[p[0], round(imgheight - p[1],2)] for p in points]
            # calculate bbox
            if calc_bbox:
                bbox = get_bbox_from_points(points)
                bbox_mode = BoxMode.XYWH_ABS # <BoxMode.XYWH_ABS: 1>
            else:
                bbox = None
                bbox_mode = None
            # Flatten the list of points and use it as segmentation data
            segmentation = [coord for point in points for coord in point]
            # create annotation
            annotation = {
                "segmentation": [segmentation],
                "area": None,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": bbox,
                "bbox_mode": bbox_mode, #"xywh", or "xyxy
                "category_id": objectId,
                "id": ann_id
            }
            # add annotation to annotations
            coco_dict['annotations'].append(annotation)
            ann_id += 1
    
    return coco_dict

def extract_polygons(image_path, extract_segment_grayscale = False):
    """
    Extract polygons from grayscale image.

    Parameters
    ----------
    image_path : str
        Path to image file.
    extract_segment_grayscale : bool
        If True, the grayscale value inside the polygon is calculated and added to the dataframe as objectID. Default: False.
        If False, it is assumed that there is only one object category and objectID is set to 1.


    Returns
    -------
    df : pandas dataframe
        Dataframe with columns ['contourId', 'objectId', 'colorId', 'polygon']

    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a threshold to get a binary image. Not needed since cv2.findContours works with grayscale images.
    #_, binary_image = cv2.threshold(image, 0, np.max(image), cv2.THRESH_BINARY)

    # Find contours and value 
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # cv2.CHAIN_APPROX_NONE
    # cv2.RETR_EXTERNAL: retrieves only the extreme outer contours)
    #  cv2.CHAIN_APPROX_SIMPLE (compresses horizontal, diagonal, and vertical segments and leaves only their end points) and cv2.CHAIN_APPROX_NONE (stores all the contour points)

    lst = []
    for i, contour in enumerate(contours): 
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if extract_segment_grayscale:
            # Create a mask for this polygon
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            # Calculate the average grayscale value inside the polygon
            #mean_val = round(cv2.mean(image, mask=mask)[0])
            # calculate the median grayscale value inside the polygon
            median_val = int(np.median(image[mask == 255]))
        else:
            median_val = 1
        lst.append([i, median_val, polygon.reshape(-1, 2)])

    return pd.DataFrame(lst, columns = ['contourId', 'objectId', 'polygon'])

def convert(path_anno, path_img, format_img = 'tif', outfname_coco = None, extract_segment_grayscale = False):
    """
    Converts raster annotations to to a COCO annotation json file for segmentation.

    Parameters
    ----------
    path : str
        Path to raster annotation files.
    path_img : str
        Path to image files. Used to extract metadata needed and to add information to json.
        Image files must have the same name as the z-coordinate at the end.
    format_img : str
        Format of image files. Default: 'png'.
    outfname_coco : str
        Path to output json file. If None, a json file with the same name as the model file will be created.
    """

    # get image metadata such as imgheight, imgwidth and name of images in path_img
    img_files = [f for f in os.listdir(path_img) if f.endswith('.' + format_img.lower())]
    # sort files by z-coordinate by extracting only numeric characters from end of filename
    img_files = sorted(img_files, key = lambda x: int(extract_numeric_characters(x.split('.')[0])))
    # get annotation files
    anno_files = [f for f in os.listdir(path_anno) if f.endswith('.' + format_img.lower())]
    anno_files =  sorted(anno_files, key = lambda x: int(extract_numeric_characters(x.split('.')[0])))
    # extract z-coordinate from filename as integer
    z_img_list = [int(extract_numeric_characters(f.split('.')[0])) for f in anno_files]
    # get image height and width of first image and assume all images have same size
    imgwidth, imgheight = get_image_size(os.path.join(path_img, img_files[0]))

    # get df from annotation files
    print("extracting polygons from annotation images...")
    for i, f in enumerate(anno_files):
        dfsel = extract_polygons(os.path.join(path_anno, f), extract_segment_grayscale = extract_segment_grayscale)
        dfsel['z'] = z_img_list[i]
        if i == 0:
            df = dfsel
        else:
            df = pd.concat([df, dfsel])

    z_list = df['z'].unique().tolist()
        
    # check if z-coordinates of model and images match
    assert len(z_list) == len(z_img_list), 'Error: Number of z-coordinates in model and images in image folder do not match.'

    coco_dict = None
    # loop over all z-coordinates and add image and annotations to coco_dict
    print('Generating COCO dictionary...')
    for i, z in enumerate(z_list):
        df_z = df[df['z'] == z].copy()
        coco_dict = generate_coco_dict(df_z, os.path.join(path_img,img_files[i]), coco_dict, imgheight, imgwidth, imgformat = format_img)
        
    # save as json file
    print('Generating json file...')
    if outfname_coco is None:
        outfname_coco = os.path.join(path_anno, 'annotations.json')
    with open(outfname_coco, 'w') as f:
        json.dump(coco_dict, f, cls=NumpyEncoder)
        
    # validate COCO json file
    validate_coco_json(outfname_coco)
    print('COCO json file is valid. Saved as ' + outfname_coco)


def main():
    parser = argparse.ArgumentParser(description='Convert raster annotations to COCO json format.')
    parser.add_argument('--path_annotation', type=str, required=True, help='Path to raster annotation files.')
    parser.add_argument('--path_img', type=str, required=True, help='Path to image files. Used to extract metadata needed and to add information to json.')
    parser.add_argument('--format_img', type=str, default='tif', help='Format of image files. Default: tif.')
    parser.add_argument('--outfname_coco', type=str, default=None, help='Path to output json file. If None, a json file with the same name as the model file will be created.')
    parser.add_argument('--extract_segment_grayscale', action='store_true', help='If called, the grayscale value inside the polygon is calculated and added to the dataframe as objectID. Default: False.')
    args = parser.parse_args()
    convert(args.path_annotation, args.path_img, args.format_img, args.outfname_coco, args.extract_segment_grayscale)

if __name__ == '__main__':
    main()