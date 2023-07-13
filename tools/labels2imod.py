"""
Converts polygon label files in json format to IMOD model files (.mod).

Requirements:
------------
- IMOD installed (tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- pandas



Example
-------
import labels2imod
inpath = '/path/to/json/files/'
labels2imod.convert_json(inpath)
"""

import os
import pandas as pd
import subprocess

def convert_json(inpath, outfname_mod = None, save_points = True):
    """
    Reads in all json files within folder, converts the polygons and object labels into a list of points and exports as imod .mod model
    The points are saved in a text file with the following columns: ['objectId', 'contourId', 'x', 'y', 'z'].
    The coordinate z is inferred from the image and json name (see below).

    The final imod model is saved as .mod file and can be opened in 3dmod.

    The json files need to include the following information:
    - 'imageHeight' : height of the image
    - 'shapes' : list of objects with points and labels
    Each shape has the following information:
        - 'label' : list of labels, one for each polygon
        - 'points' : list of points for each polygon
    

    Parameters
    ----------
    inpath : str
        Path to json files. Json files must be named as follows: 'image_z<z>.json' or image_<z>.json where <z> is the z-coordinate of the image.
    outfname_mod : str
        Path + filename for output file in imod .mod format. If None, file is written to same path as json files.
    save_points : bool
        If True, points are saved as txt file in same path as json files.

    Returns
    -------
    df : pandas dataframe
    """
    # get all json files
    json_files = [f for f in os.listdir(inpath) if f.endswith('.json')]
    # sort files by z-coordinate
    json_files = sorted(json_files, key = lambda x: int(x.split('_')[1].split('.')[0].replace('z', '')))
    # extract z-coordinate from filename as integer
    z_list = [int(f.split('_')[1].split('.')[0].replace('z', '')) for f in json_files]

    # define dataframe to save all points
    df_all = pd.DataFrame(columns = ['objectId', 'contourId', 'x', 'y', 'z'])
    #image_list = []
    for i, f in enumerate(json_files):
        # read in json files
        with open(os.path.join(inpath, f), 'r') as f:
            data = json.load(f)
        # get image name
        # image_list.append(data["imagePath"])
        # get z-coordinate
        z = z_list[i]
        # get image height for mirroring y-coordinates
        image_height = data['imageHeight']
        # get shapes
        shapes = data['shapes']
        # loop over shapes
        for j, shape in enumerate(shapes):
            label = shape['label']
            # get points
            points = shape['points']
            # create dataframe
            df = pd.DataFrame(columns = ['objectId', 'contourId', 'x', 'y', 'z'])
            df['objectId'] = [label] * len(points)
            df['contourId'] = [z] * len(points)
            df['x'] = [p[0] for p in points]
            df['y'] = [image_height - p[1] for p in points]
            df['z'] = [z] * len(points)
            # combine with df_all
            df_all = pd.concat([df_all, df], ignore_index = True)

    # Save points as txt file
    fname_txt = os.path.join(inpath, 'model_points.txt')
    df_all.to_csv(fname_txt, sep='\t', index = False, header = False)

    # convert point file to imod model
    if outfname_mod is None:
        outfname_mod = os.path.join(inpath, 'IMOD_model.mod')
    imod_cmd = f'point2model {fname_txt} {outfname_mod} -origin 0,0,{min(z_list)}' #-scale 1,1,1'
    try:
        print('Creating IMOD model from point file...')
        subprocess.run([imod_cmd], shell = True, check = True)
    except subprocess.CalledProcessError:
        print('Error with model2point command:' + imod_cmd)
    # check if file exists
    if not os.path.isfile(outfname_mod):
        print(f'Error: point2model failed. No output file found.')
    print(f'Model saved to {outfname_mod}.')
    if not save_points:
        os.remove(fname_txt)        
