"""
Convert IMOD model (.mod) to a COCO json file with annotations for polygon segmentation.

Usage:
python imod2coco.py -i IMOD_model.mod -p /path/to/image/files/ -f png

Required Arguments:
-i, --input: Filename for IMOD model file
-p, --path_img: Path to image files
-f, --format_img: Format of image files, default is png

Optional Arguments:
-o, --output: Filename for output json file, default is None (same name as model file)

Note: image files names must include the z-coordinate number at the end after '_'.

Requirements:
------------
- IMOD e'(tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- pandas
- json
- PIL

Example Python
--------------------
import imod2coco
infname_mod = 'IMOD_model.mod'
path_img = '/path/to/image/files/'
imod2coco.convert_to_coco(infname_mod, path_img, format_img = 'png')


Author: Sebastian Haan, The University of Sydney, 2023
"""

import os
import re
import argparse
import datetime
import pandas as pd
import numpy as np
import subprocess
from PIL import Image
import json


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


def load_imod_model(infname, delete_temp_modeltxt = True):
    """
    Converts IMOD model (.mod) to pandas dataframe

    Parameters
    ----------
    infname : str
        Path to IMOD model file.
    delete_temp_modeltxt : bool
        If True, temporary file created by model2point is deleted after loading into pandas dataframe. Default: False.

    Returns
    -------
    df : pandas dataframe
    """
    # create temporary file to save points
    outfname_temp = infname.replace('.mod', '.txt')

    # run extraction of points from imod model
    imod_cmd = f'model2point -input {infname} -output {outfname_temp} -object -ZCoordinatesFromZero'
    try:
        print('Loading IMOD model and creating temporary point file. This may take a while...')
        subprocess.run([imod_cmd], shell = True, check = True)
    except subprocess.CalledProcessError:
        print('Error with model2point command:' + imod_cmd)
        return None
    # check if file exists
    if not os.path.isfile(outfname_temp):
        print(f'Error: model2point extraction failed. No output file found.')
        return None

    # generate header names for df
    header = ['objectId', 'contourId', 'x', 'y', 'z'] # 'size', 'pointId', 'flags', 'time', 'value']

    df = pd.read_csv(outfname_temp, sep = '\s+', header = None, names = header)
    # remove duplicates
    df = df.drop_duplicates()
    # remove temporary file
    if delete_temp_modeltxt:
        os.remove(outfname_temp)
    return df

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


def generate_coco_dict(
    df, 
    imgfile, 
    coco_dict = None, 
    imgheight = None, 
    imgwidth = None, 
    imgformat = 'tif', 
    coco_info = None, 
    licenses = None):
    """
    Generate COCO dict from dataframe.
    Here the dict format follows the format of the COCO dataset.

    Parameters
    ----------
    df : pandas dataframe
        must include at least the following columns: ['objectId', 'x', 'y']
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
            points = df_obj[df_obj['contourId'] == contourId][['x', 'y']].values.tolist()
            # reverse y coordinates with respect to image height
            points = [[p[0], round(imgheight - p[1],2)] for p in points]
            # Flatten the list of points and use it as segmentation data
            segmentation = [coord for point in points for coord in point]
            # create annotation
            annotation = {
                "segmentation": [segmentation],
                "area": None,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": None,
                "category_id": objectId,
                "id": ann_id
            }
            # add annotation to annotations
            coco_dict['annotations'].append(annotation)
            ann_id += 1
    
    return coco_dict


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


def convert_to_coco(infname_mod, path_img, format_img = 'png', outfname_coco = None):
    """
    Converts IMOD model (.mod) to to a COCO annotation json file for segmentation.

    Parameters
    ----------
    infname : str
        Path to IMOD model file.
    path_img : str
        Path to image files. Used to extract metadata needed and to add information to json.
        Image files must have the same name as the z-coordinate at the end after '_'.
    format_img : str
        Format of image files. Default: 'tif'.
    outfname_coco : str
        Path to output json file. If None, a json file with the same name as the model file will be created.
    """
    # First load model into pandas dataframe
    df = load_imod_model(infname_mod)
    # get sorted z-coordinates
    z_list = sorted(df['z'].unique())

    # get image metadata such as imgheight, imgwidth and name of images in path_img
    img_files = [f for f in os.listdir(path_img) if f.endswith('.' + format_img.lower())]
    # sort files by z-coordinate by extracting only numeric characters from end of filename
    img_files = sorted(img_files, key = lambda x: int(extract_numeric_characters(x.split('_')[-1].split('.')[0])))
    # extract z-coordinate from filename as integer
    z_img_list = [int(extract_numeric_characters(f.split('_')[-1].split('.')[0])) for f in img_files]
    # get image height and width of first image and assume all images have same size
    imgwidth, imgheight = get_image_size(os.path.join(path_img, img_files[0]))

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
    outfname_coco = infname_mod.replace('.mod', '.json')
    with open(outfname_coco, 'w') as f:
        json.dump(coco_dict, f, cls=NumpyEncoder)
        
    # validate COCO json file
    validate_coco_json(outfname_coco)
    print('COCO json file is valid. Saved as ' + outfname_coco)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Convert IMOD model (.mod) to COCO annotation json file for segmentation.')
    parser.add_argument('-i', '--input', help='Filename for IMOD model file', required=True)
    parser.add_argument('-p', '--path_img', help='Path to image files', required=True)
    parser.add_argument('-f', '--format_img', help='Format of image files', default='png', choices=['tif', 'tiff', 'png', 'jpg'])
    parser.add_argument('-o', '--output', help='Filename for output json file', default=None)
    args = parser.parse_args()

    # Convert to COCO format
    fnames_json = convert_to_coco(args.input, args.path_img, format_img = args.format_img, outfname_coco = args.output)

if __name__ == '__main__':
    main()