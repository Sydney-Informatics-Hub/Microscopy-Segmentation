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
import math
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .tif images to another format.')
    parser.add_argument('-i', '--input_dir', help='Directory of .tif images', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', default='./output')
    parser.add_argument('-f', '--format', help='Output format', default='png', choices=['png', 'jpg'])
    parser.add_argument('-s', '--size', help='Size for output images', type=int)
    parser.add_argument('-c', '--color_mode', help='Color mode for output images', choices=['RGB', 'L'])
    args = parser.parse_args()

    convert_images(args.input_dir, args.output_dir, args.format, args.size, args.color_mode)
