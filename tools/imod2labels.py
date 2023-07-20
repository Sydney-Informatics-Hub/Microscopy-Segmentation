"""
Convert IMOD model (.mod) to a json file with labels for polygon and points segmentation.
For each image a json file is created and stored in the same folder as the image file.

Usage:
python imod2labels.py -i IMOD_model.mod -p /path/to/image/files/ -f png

Required Arguments:
-i, --input: Filename for IMOD model file
-p, --path_img: Path to image files
-f, --format_img: Format of image files, default is png

Note: image files names must include the z-coordinate number at the end after '_'.

Supported JSON segmentation formats:
Anylabeling: https://pypi.org/project/anylabeling/
YOLO
SAM

Requirements:
------------
- IMOD (tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- pandas
- json
- PIL

Example Python
--------------------
import imod2labels
infname_mod = 'IMOD_model.mod'
path_img = '/path/to/image/files/'
fnames_json = imod2labels.convert_to_anylabeling(infname_mod, path_img, format_img = 'png')


Author: Sebastian Haan, The University of Sydney, 2023
"""

import os
import re
import argparse
import pandas as pd
import subprocess
from PIL import Image
import json
from jsonschema import validate, ValidationError


def validate_json(json_data, json_schema):
    """
    Validates json file against json schema.

    Parameters
    ----------
    json_data : json file
        Json file to be validated.
    json_schema : json file
        Json schema to validate against.

    Returns
    -------
    bool
        True if json file is valid, False otherwise.
    """

    # Load your JSON data
    with open(json_data, 'r') as f:
        data = json.load(f)

    # Load your schema
    with open(json_schema, 'r') as f:
        schema = json.load(f)

    # Validate your data
    try:
        validate(instance=data, schema=schema)
        #print("JSON data is valid")
    except ValidationError as e:
        print(f"JSON data is invalid: {e}")
        return False
    return True


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

def generate_json_anylabeling(df, imgfile, imgheight = None, imgwidth = None, imgformat = 'tif'):
    """
    Generate json file from dataframe.
    Here the json file follows the format of the Anylabeling tool.
    The json file is saved in the same folder as the image file.

    Parameters
    ----------
    df : pandas dataframe
        must include at least the following columns: ['objectId', 'x', 'y']
    imgfile : str
        Path to image file.
    imgheight : int
        Height of image in pixels. If None, extracted from image file.
    imgwidth : int
        Width of image in pixels. If None, extracted from image file.
    imgformat : str
        Format of image file. Default: 'tif'.
    
    Returns
    -------
    json : json filename.
    """
    # if imgheight or imgwidth are not provided, get them from image file
    if (imgheight is None) or (imgwidth is None):
        imgwidth, imgheight = get_image_size(imgfile)

    # get name of image file without path
    imgfile_name = os.path.basename(imgfile)

    shapes = []
    # get unique objectIds
    objectIds = df['objectId'].unique()
    for objectId in objectIds:
        # get all points for this objectId
        df_obj = df[df['objectId'] == objectId].copy()
        # loop over all contourIds
        contourIds = df_obj['contourId'].unique()
        for contourId in contourIds:
            # get all points for this contourId
            points = df_obj[df_obj['contourId'] == contourId][['x', 'y']].values.tolist()
            # reverse y coordinates with respect to image height
            points = [[p[0], round(imgheight - p[1],2)] for p in points]
            # create shape
            shape = {
                'label' : str(objectId), 
                "text" : "", 
                'points' : points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}}
            # add shape to shapes
            shapes.append(shape)

    # create json dictionary
    json_dict = {
         "version": "0.3.3",
         "flags": {},
        "shapes": shapes,
        "imagePath": imgfile_name,
        "imageData": None,
        'imageHeight' : imgheight, 
        'imageWidth' : imgwidth,
        "text" : "",
        }
    # save json file
    outfname_json = imgfile.replace('.' + imgformat.lower(), '.json')
    with open(outfname_json, 'w') as f:
        json.dump(json_dict, f, indent = 4)
    return outfname_json

def convert_to_anylabeling(infname_mod, path_img, format_img = 'png'):
    """
    Converts IMOD model (.mod) to to a json file with labels and points according to AnyLabeling format.
    see: https://pypi.org/project/anylabeling/
    For each image a json file is created and stored in the same folder as the image file.

    Parameters
    ----------
    infname : str
        Path to IMOD model file.
    path_img : str
        Path to image files. Used to extract metadata needed and to add information to json.
        Image files must have the same name as the z-coordinate at the end after '_'.
    format_img : str
        Format of image files. Default: 'tif'.
    
    Returns
    -------
    outfnames : list of json filenames that were created.
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

    # generate json file for each z
    outfnames = []
    for i, z in enumerate(z_list):
        df_z = df[df['z'] == z].copy()
        
        # generate json file
        outfname_json = generate_json_anylabeling(df_z, os.path.join(path_img,img_files[i]), imgheight, imgwidth, imgformat = format_img)
        
        # validate json file
        valid = validate_json(outfname_json, 'json_schema_anylabeling.json')
        if valid:
            print(f'Created Json file {outfname_json}')
        else:
            print(f'Error: Json file {outfname_json} is not valid.')
        outfnames.append(outfname_json)

    return outfnames

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Convert IMOD model (.mod) to a json file with labels for polygon and points segmentation.')
    parser.add_argument('-i', '--input', help='Filename for IMOD model file', required=True)
    parser.add_argument('-p', '--path_img', help='Path to image files', required=True)
    parser.add_argument('-f', '--format_img', help='Format of image files', default='png', choices=['tif', 'tiff', 'png', 'jpg'])
    args = parser.parse_args()

    # convert to anylabeling format
    fnames_json = convert_to_anylabeling(args.input, args.path_img, format_img = args.format_img)

if __name__ == '__main__':
    main()