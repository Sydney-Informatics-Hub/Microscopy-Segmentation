"""Python tool for conversion and processing of .tif images to another format.

Usage:
python tif2x.py --input_dir /path/to/tif/images --output_dir /path/to/output/images --format jpg --size 500 --color_mode RGB

Required Arguments:
--input_dir: Directory of .tif images
--output_dir: Output directory

Optional Arguments:
--format: Output format (e.g. jpg, png), default is png
--size: Size for output images, default is None (not changed)
--color_mode: Color mode for output images (e.g. RGB, L), default is None (not changed)
"""

import os
import argparse
from PIL import Image

def convert_images(input_dir, output_dir, output_format='png', size=None, color_mode=None):
    """
    Convert .tif images to another format.

    Args:
        input_dir (str): Directory of .tif images
        output_dir (str): Output directory
        output_format (str): Output format (e.g. jpg, png), default is png
        size (int): Size for output images, default is None (not changed)
        color_mode (str): Color mode for output images (e.g. RGB, L), default is None (not changed)

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            img = Image.open(os.path.join(input_dir, filename))

            # Handle 'I;16B' image mode
            if img.mode == 'I;16B':
                img = img.convert('I')
            
            if size:
                img = img.resize((size, size))
            
            if color_mode:
                if color_mode == 'RGB':
                    img = img.convert('RGB')
                elif color_mode == 'L':
                    img = img.convert('L')

            basename = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{basename}.{output_format}")
            img.save(output_file)
            print(f"Image {output_file} has been saved.")


def split_image_into_tiles(infname_img, tile_width, tile_height, overlap, outpath=None):
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
    overlap : bool
        If True, tiles are allowed to overlap.
        If False, tiles are not allowed to overlap and will be resampled to tile_width and tile_height if necessary.
    outpath : str
        Output path for tiles
        If none, tiles are saved in the same directory as the input image
    """
    # read image
    img = Image.open(infname_img)
    img_width, img_height = img.size

    # If outpath is not provided, save in the same directory as the input image
    os.makedirs(outpath, exist_ok=True)

    # Create the output directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Calculate overlap ratio
    if overlap:
        overlap_ratio_width = (tile_width - (img_width % tile_width) / (img_width // tile_width)) / tile_width
        overlap_ratio_height = (tile_height - (img_height % tile_height) / (img_height // tile_height)) / tile_height
        print(f"Overlap ratio (width): {overlap_ratio_width:.2f}")
        print(f"Overlap ratio (height): {overlap_ratio_height:.2f}")

        num_tiles_x = (img_width - tile_width) // (tile_width // 2) + 1
        num_tiles_y = (img_height - tile_height) // (tile_height // 2) + 1
    else:
        num_tiles_x = img_width // tile_width
        num_tiles_y = img_height // tile_height

    # Get the image basename and extension
    base_name = os.path.basename(infname_img)
    name, ext = os.path.splitext(base_name)

    # Split the image into tiles and save them
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            if overlap:
                left = i * (tile_width // 2)
                upper = j * (tile_height // 2)
            else:
                left = i * tile_width
                upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # Ensure the tile coordinates are within the image boundaries
            if right > img_width or lower > img_height:
                continue

            tile = img.crop((left, upper, right, lower))

            # If no overlap and tile size is different, resize the tile
            if not overlap and (tile.size != (tile_width, tile_height)):
                tile = tile.resize((tile_width, tile_height))

            tile_filename = os.path.join(outpath, f"{name}_{i}_{j}{ext}")
            tile.save(tile_filename)
    print(f"Tiles saved in {outpath}")

def test_split_image_into_tiles():
    infname_img = '../testdata/sample_anylabeling/HM25_001.png'
    tile_width = 640
    tile_height = 640
    overlap=False
    outpath = '../testdata/test_tiles'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .tif images to another format.')
    parser.add_argument('-i', '--input_dir', help='Directory of .tif images', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', default='./output')
    parser.add_argument('-f', '--format', help='Output format', default='png', choices=['png', 'jpg'])
    parser.add_argument('-s', '--size', help='Size for output images', type=int)
    parser.add_argument('-c', '--color_mode', help='Color mode for output images', choices=['RGB', 'L'])
    args = parser.parse_args()

    convert_images(args.input_dir, args.output_dir, args.format, args.size, args.color_mode)
