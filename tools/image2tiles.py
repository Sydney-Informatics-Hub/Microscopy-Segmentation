"""
Python tool for splitting and images into multiple tiles.
User has the option to allow tiles to overlap or not. The optimal overlap ratio is automatically calculated.
If no overlap is allowed, the image is resampled to the nearest multiple of the tile size.

Usage:
python image2tiles.py --input_dir /path/to/images --output_dir /path/to/output --tile_width 640 --tile_height 640 --max_overlap_ratio 0.2

Required Arguments:
--input_dir: Directory of images (accepted formats: .png, .jpg, .jpeg, .tif, .tiff)
--tile_width: Tile width (pixels)
--tile_height: Tile height (pixels)

Optional Arguments:
--output_dir: Output directory
--max_overlap_ratio: Maximum overlap ratio. If larger than 0, tiles are allowed to overlap. 
    Otherwise, tiles are not allowed to overlap and will be resampled to tile_width and tile_height

Author: Sebastian Haan
"""

import os
import argparse
import math
from PIL import Image

def split_image_into_tiles(infname_img, tile_width, tile_height, max_overlap_ratio, outpath=None):
    """
    Split images into tiles. Tiles are named as follows: <basename_img>_<x>_<y>.<ext>

    Parameters
    ----------
    infname_img : str
        Input image filename
    
    tile_width : int
        Tile size width
    tile_height : int
        Tile size height
    max_overlap_ratio: 
        If > 0, tiles are allowed to overlap. Overlap ratio is automatically calculated
        If = 0 , tiles are not allowed to overlap and will be resampled to tile_width and tile_height if necessary.
    outpath : str
        Output path for tiles
        If none, tiles are saved in the same directory as the input image
    """
    # Open the image
    img = Image.open(infname_img)
    img_width, img_height = img.size

    # Create the output directory if it doesn't exist
    os.makedirs(outpath, exist_ok=True)
    
    if max_overlap_ratio == 0 and (img.size != (tile_width, tile_height)):
        # Sample image to nearest multiple of tile size
        num_tiles_x = round(img_width /tile_width) 
        num_tiles_y = round(img_height /tile_height)

        img_width_new = num_tiles_x * tile_width
        img_height_new = num_tiles_y * tile_height
        img = img.resize((img_width_new, img_height_new))
        img_width, img_height = img.size
        step_size_width = tile_width
        step_size_height = tile_height

    elif img.size == (tile_width, tile_height):
        print('Tile size is the same as image size. No need to resize.')
        return [infname_img]

    else:
        # make tiles with overlap
        # Calculate the number of tiles
        num_tiles_x = math.ceil(img_width /tile_width) 
        num_tiles_y =math.ceil(img_height /tile_height)

        # Calculate the remaining space to be covered
        remaining_width = (num_tiles_x * tile_width) - img_width
        remaining_height = (num_tiles_y * tile_height) - img_height 

        # Calculate the overlap required to cover the remaining space
        overlap_width = int(remaining_width / (num_tiles_x - 1)) if num_tiles_x > 1 else 0
        overlap_height = int(remaining_height / (num_tiles_y - 1)) if num_tiles_y > 1 else 0

        # Ensure the overlap does not exceed the max_overlap_ratio
        overlap_ratio_width = overlap_width / tile_width
        overlap_ratio_height = overlap_height / tile_height
        assert overlap_ratio_width <= max_overlap_ratio, f"Overlap ratio (width) {overlap_ratio_width:.2f} exceeds max overlap ratio {max_overlap_ratio:.2f}. Adjust overlap_ratio or set to zero for resampling."
        assert overlap_ratio_height <= max_overlap_ratio, f"Overlap ratio (height) {overlap_ratio_height:.2f} exceeds max overlap ratio {max_overlap_ratio:.2f}. Adjust overlap_ratio or set to zero for resampling."

        print(f"Optimal overlap ratio (width, height): {overlap_ratio_width:.2f}, {overlap_ratio_height:.2f}")

        # Calculate step size based on the overlap
        step_size_width = tile_width - overlap_width
        step_size_height = tile_height - overlap_height

    # Get the image basename and extension
    base_name = os.path.basename(infname_img)
    name, ext = os.path.splitext(base_name)

    # Split the image into tiles and save them
    filename_list = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            left = i * step_size_width
            upper = j * step_size_height
            # Adjust for pixel rounding at the edges
            if left + tile_width > img_width:
                left = img_width - (tile_width)
            if upper + tile_height > img_height:
                upper = img_height - (tile_height)
            right = min(left + tile_width, img_width)  # Ensure right doesn't exceed image width
            lower = min(upper + tile_height, img_height)  # Ensure lower doesn't exceed image height
                
            tile = img.crop((left, upper, right, lower))
            tile_filename = os.path.join(outpath, f"{name}_{i}_{j}{ext}")
            tile.save(tile_filename)
            filename_list.append(tile_filename)

    #print(f"Tiles saved in {outpath}")
    return filename_list


def tile_images_in_folder(image_folder, tile_width, tile_height, max_overlap_ratio, outpath=None):
    """
    Split all images in a folder into tiles.

    Parameters
    ----------
    image_folder : str
        Input image folder
    tile_width : int
        Tile size width
    tile_height : int
        Tile size height
    max_overlap_ratio:
        If > 0, tiles are allowed to overlap. Overlap ratio is automatically calculated
        If = 0 , tiles are not allowed to overlap and will be resampled to tile_width and tile_height if necessary.
    outpath : str
        Output path for tiles
        If none, tiles are saved in the same directory as the input image
    """
    # Get all images in the folder with file extensions .png, .jpg, .jpeg, .tif, .tiff
    image_list = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_list.extend([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(ext)])

    # Split each image into tiles use tqdm to show progress bar
    outfname_list = []
    for infname_img in image_list:
        print(f"Tiling {infname_img}...")
        outfname_list += split_image_into_tiles(infname_img, tile_width, tile_height, max_overlap_ratio, outpath)
    return


def test_split_image_into_tiles():
    infname_img = '../testdata/sample_anylabeling/HM25_001.png'
    tile_width = 640
    tile_height = 640
    max_overlap_ratio =0.2
    outpath = '../testdata/test_tiles_overlapmax'
    fnames = split_image_into_tiles(infname_img, tile_width, tile_height, max_overlap_ratio, outpath)
    #tile_images_in_folder('../testdata/sample_anylabeling', 640, 640, 0.2, '../testdata/test_tiles_overlapmax')



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split images into tiles with or without overlap.')
    parser.add_argument('-i', '--input_dir', help='Directory of images (accepted formats: .png, .jpg, .jpeg, .tif, .tiff)', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', default=None)
    parser.add_argument('-tw', '--tile_width', help='Tile width (pixels)', type=int, required=True)
    parser.add_argument('-th', '--tile_height', help='Tile height (pixels)', type=int, required=True)
    parser.add_argument('-ratio', '--max_overlap_ratio', help='Maximum overlap ratio. If larger than 0, tiles are allowed to overlap.\
        Otherwise, tiles are not allowed to overlap and will be resampled to tile_width and tile_height', type=float, default=0)
    args = parser.parse_args()

    # Split images into tiles
    tile_images_in_folder(args.input_dir, args.tile_width, args.tile_height, args.max_overlap_ratio, args.output_dir)


if __name__ == '__main__':
    main()