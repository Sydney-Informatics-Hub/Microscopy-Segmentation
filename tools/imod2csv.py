"""
Read IMOD model to pandas dataframe and saving to csv file.

Requirements:
------------
- IMOD installed (tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- pandas


Example
-------
import imod2csv
infname = 'IMOD_model.mod'
df = imod2csv.load_model(infname)
"""

import os
import pandas as pd
import subprocess

def load_model(infname, outfname_csv = None, args_model2point = '-ZCoordinatesFromZero'):
    """
    Reads in IMOD model (.mod) to pandas dataframe and saves it as csv file.

    Parameters
    ----------
    infname : str
        Path to IMOD model file.
    outfname_csv : str
        Path + filename for output file in csv format. If None, file is written to same path as csv.
    args_model2point : str
        Additional arguments for model2point command. See IMOD documentation for details.
        (https://bio3d.colorado.edu/imod/doc/man/model2point.html)

    Returns
    -------
    df : pandas dataframe
    """
    # create temporary file to save points
    outfname_temp = infname.replace('.mod', '.txt')

    # run extraction of points from imod model
    imod_cmd = f'model2point -input {infname} -output {outfname_temp} -object {args_model2point}'
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
    os.remove(outfname_temp)

    if outfname_csv is not None:
        # check if output directory exists
        outdir = os.path.dirname(outfname_csv)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok = True)
    else:
        outfname_csv = infname.replace('.mod', '.csv')
    df.to_csv(outfname_csv, index = False)  
    print('Loaded model saved to ' + outfname_csv + '.')

    return df
