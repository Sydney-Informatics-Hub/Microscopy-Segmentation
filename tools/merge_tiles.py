# Merge image tiles into a single image

import argparse
import os
from PIL import Image

def merge(tile_dir, output_file, resize=None):
    """
    Merge tiles into a single image.
    Tiles need to be labeled in the format <BASENAME>_i_j.<EXT> , 
    where i and j are ith and jth tile number in x and y direction (starting from 0).
    All tiles must have the same size.

    Accepted extensions are .png, .jpg., .jpeg, .tif, .tiff.

    Parameters
    ----------
    tile_dir : str
        Directory of tiles
    output_file : str
        Output path + filename.
    resize : tuple
        Resize output image to this size (width, height)
    """

    # accepted input image formats
    accepted_formats = ["png", "jpg", "jpeg", "tif", "tiff"]
    image_list = [f for f in os.listdir(tile_dir) if f.split(".")[-1].lower() in accepted_formats]

    # check that images conform to naming convention
    image_list_ok = []
    x_list = []
    y_list = []
    for fname in image_list:
        basename = fname.split(".")[0]
        basename_split = basename.split("_")
        # check that  i and j in _i_j of basename_split and check if i and j are integers)
        try:
            x_list.append(int(basename_split[-2]))
            y_list.append(int(basename_split[-1]))
            image_list_ok.append(fname)
        except:
            raise ValueError(f"Tile name {fname} does not conform to naming convention: <BASENAME>_i_j.<EXT>. Skipping")
    
    # get number of tiles in x and y direction
    x_tiles = len(x_list)
    y_tiles = len(y_list)

    # check that no tile is missing
    if x_tiles != len(set(x_list)) or y_tiles != len(set(y_list)):
        raise ValueError(f"Number of tiles in x and y direction ({x_tiles}, {y_tiles}) does not match number of tiles in folder ({len(image_list_ok)}).")

    # get tile size
    tile_size = Image.open(os.path.join(tile_dir, image_list_ok[0])).size
        
    # merge tiles into a single image
    img_merged = Image.new('RGB', (x_tiles * tile_size[0], y_tiles * tile_size[1]))
    for fname in image_list_ok:
        basename = fname.split(".")[0]
        basename_split = basename.split("_")
        x = int(basename_split[-2])
        y = int(basename_split[-1])
        img_tile = Image.open(os.path.join(tile_dir, fname))
        # check that tile has same tile_size
        if img_tile.size != tile_size:
            raise ValueError(f"Tile {fname} has size {img_tile.size}, but should have same size as first tile {tile_size}.")
        img_merged.paste(img_tile, (x * tile_size[0], y * tile_size[1]))

    if resize is not None:
        img_merged = img_merged.resize(resize)

    # save output image
    img_merged.save(output_file)