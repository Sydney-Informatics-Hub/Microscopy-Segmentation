"""
Converts Anylabeling json annotation files to multistack tiff images.

Example:
    python anylabeling2tif.py --path_json /path/to/json/files --path_out /path/to/output/

Features:
- Supports multiple labels per image
- Supports multiple polygons per label
- Supports z-coordinates for 3D image stacks
- Save color-coded images with labels as tiff images


The Anylabeling json files need to include the following information:
- 'imageHeight' : height of the image
- 'imageWidth' : width of the image
- 'imagePath' : path to image
- 'shapes' : list of objects with points and labels
Each shape has the following information:
    - 'label' : list of labels, one for each polygon
    - 'points' : list of points for each polygon
    - 'shape_type' : type of shape, e.g. 'polygon'

Author: Sebastian Haan
"""

import os
import argparse
import pandas as pd
import json
import numpy as np
import rasterio

import cv2
import tifffile as tiff
import imageio



def extract_after_last_z(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    last_z_index = basename.rfind('z')
    if last_z_index == -1: # 'z' not found in basename
        return ""
    return basename[last_z_index + 1:]


def anylabeling2df(path_json, outfname = None, get_z = False):
    """
    Write annotations in anylabeling json files to pandas dataframe.

    Parameters
    ----------
    path_json : str
        Path to json files. Json files must be named as follows: 'image_z<z>.json' or image_<z>.json where <z> is the z-coordinate of the image.
    outfname : str, optional
        Path + filename for output file in imod .mod format. If None, file is written to same path as json files.
    get_z : bool
        If True, z-coordinate is extracted from filename. If False, no z-coordinate is extracted.

    Returns
    -------
    df : pandas dataframe
    json_files : list
    """
    # get all json files
    json_files = [f for f in os.listdir(path_json)   if f.endswith('.json')]

    if get_z:
        # sort files by z-coordinate
        json_files = sorted(json_files, key = lambda x: int(extract_after_last_z(x)))
        # extract z-coordinate from filename as integer
        z_list = [int(extract_after_last_z(f)) for f in json_files]

    # define dataframe to save all points
    df_all = pd.DataFrame(columns = ['label', 'points', 'shapetype', 'imagePath', 'imageHeight', 'imageWidth', 'z'])
    #image_list = []
    for i, f in enumerate(json_files):
        # read in json files
        with open(os.path.join(path_json, f), 'r') as f:
            data = json.load(f)
        # get image name
        # image_list.append(data["imagePath"])
        # get z-coordinate
        if get_z:
            z = z_list[i]
        else:
            z = None
        # get shapes
        shapes = data['shapes']
        # loop over shapes
        for j, shape in enumerate(shapes):
            label = shape['label']
            # get points
            points = shape['points']
            # create dataframe
            df = pd.DataFrame(columns = ['label', 'points', 'shapetype', 'imagePath', 'imageHeight', 'imageWidth', 'z'])
            df['points'] = [points]#[[item for sublist in points for item in sublist]]
            df['label'] = label
            df['shapetype'] = shape['shape_type']
            df['imagePath'] = data['imagePath']
            df['imageHeight'] = data['imageHeight']
            df['imageWidth'] = data['imageWidth']
            df['z'] = z
            # combine with df_all
            df_all = pd.concat([df_all, df], ignore_index = True)

        # Save points as txt file
    df_all.to_csv(outfname, index = False, header = False)
    return df_all, json_files



def random_color():
    """Generate a random color."""
    return (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))


def distinct_color(num_labels, index):
    """
    Generate a distinct color from a spectrum.

    Parameters
    ----------
    num_labels : int
        Number of labels.
    index : int
        Index of label.

    Returns
    -------
    tuple
        RGB color.
    """
    hue = 255 * index / num_labels
    saturation = 255
    value = 255
    hsv_color = np.uint8([[[hue, saturation, value]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, rgb_color))

def write_polygons_to_tiff(df, outpath):
    """
    Write dataframe labels and polygons to tiff images.
    Generate a unique color for each label. 

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing labels, polygons, and image information.
    outpath : str
        Path to output directory.

    Returns
    -------
    outfname_list : list
        List of output filenames.
    dict_colors : dict
        Dictionary containing label and color information.
    """
    unique_labels = df['label'].unique()
    #color_map = {label: (i+1)*10 for i, label in enumerate(unique_labels)}  # Assigning a unique grayscale value for simplicity
    num_labels = len(unique_labels)
    color_map = {label: distinct_color(num_labels, i) for i, label in enumerate(unique_labels)} 

    outfname_list = []
    dict_colors = {}

    for imagePath, sub_df in df.groupby('imagePath'):
        # Create a blank image
        image_data = np.zeros((int(sub_df['imageHeight'].values[0]), int(sub_df['imageWidth'].values[0]), 3), dtype=np.uint8)

        # For each polygon in the sub-DataFrame, draw the polygon onto the blank image using OpenCV
        for idx, row in sub_df.iterrows():
            pts = np.array(row['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(image_data, [pts], color_map[row['label']])
            # save color for each label in dict_colors. Save only if not existent yet
            if row['label'] not in dict_colors.keys():
                dict_colors[row['label']] = color_map[row['label']]
        
        # Save the image as TIFF using tifffile
        outfname = os.path.basename(imagePath).replace('.tif', '_mask.tif')
        output_path = os.path.join(outpath, outfname)
        tiff.imwrite(output_path, image_data)
        outfname_list.append(output_path)

    return outfname_list, dict_colors



def convert_to_stack_tif(filenames, output_filename):
    """
    Convert a stack of images in a directory to a single 3D TIFF file.

    Parameters:
    - filenames: list of input filenames
    - output_filename: The name of the output 3D TIFF file.
    """
    
    #images = [imread(os.path.join(directory, image_file)) for image_file in files]
    images = [imageio.v2.imread(image_file) for image_file in filenames]

    # Save the list of images as a single 3D TIFF
    imageio.volwrite(output_filename, images)



def anylabeling_to_tif(path_json, outpath_tif, get_z = False):
    """
    Write anylabeling json files to tiff images.
    Tiff images are stacked into a single tiff file and saved to outpath_tif.
    Color dictionary is saved to outpath_tif.

    Parameters
    ----------
    path_json : str
        Path to json files. 
    outpath_tif : str
        Path to output directory.
    get_z : bool, optional
        If True, z-coordinate is extracted from json filename. The default is False.
        If True the json files must be named as follows: 'image_z<z>.json' or image_<z>.json where <z> is the z-coordinate of the image.

    Returns
    -------
    None 
    """
    # Convert and merge anylabeling json files to pandas dataframe
    df, json_files = anylabeling2df(path_json, outfname = None, get_z = get_z)

    # Write labels and  polygons to tiff images
    outfname_list, dict_colors = write_polygons_to_tiff(df, path_json)

    # stack images
    convert_to_stack_tif(outfname_list, os.path.join(outpath_tif, 'polygons_stacked.tif'))
    
    # Delete temporary tiff files
    for f in outfname_list:
        os.remove(f)

    # Save color dictionary
    with open(os.path.join(outpath_tif, 'polygons_stacked_color_dict.json'), 'w') as fp:
        json.dump(dict_colors, fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert anylabeling json files to tiff images.')
    parser.add_argument('--path_json', type=str, help='Path to json files. For 3D, json files must be named as follows: "image_z<z>.json" or "image_<z>.json" where <z> is the z-coordinate of the image.')
    parser.add_argument('--path_out', type=str, help='Path to output directory.')
    parser.add_argument('--get_z', action='store_true', default=False, help='If True, z-coordinate is extracted from json filename. The default is False.')
    args = parser.parse_args()

    anylabeling_to_tif(args.path_json, args.path_out, args.get_z)