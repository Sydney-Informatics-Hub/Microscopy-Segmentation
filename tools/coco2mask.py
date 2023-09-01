#!/usr/bin/env python

"""
Convert COCO JSON annotations to masks
Each mask is saved as a TIFF file with the category id and image_name in the filename.

Example usage:
python coco2mask.py --coco_path /path/to/coco.json --save_dir /path/to/save_dir --category_ids 1 2 3 --image_format tiff

For reverse conversion, see mask2coco.py
https://github.com/HaiyangPeng/mask-to-coco-json ?

"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict


def coco_to_mask(coco_file, output_dir, category_ids, image_format='png'):
    """
    Convert COCO JSON annotations to masks

    Args:
        coco_file (str): Path to the COCO JSON file
        output_dir (str): Directory to save the mask results
        category_ids (list): List of category ids to extract
        image_format (str): Image format to save the masks
            options: 'tiff', 'png', 'jpg'

    """
    print("Category ids:", category_ids)
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load COCO JSON annotations
    with open(coco_file, 'r') as f:
        annotations = json.load(f)

    # Create a mapping from image id to filename
    id_to_filename = {image['id']: image['file_name'] for image in annotations['images']}

    # Group annotations by image id
    annotations_by_image = defaultdict(list)
    for annotation in annotations['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)

    # Process each image
    for image_id, image_annotations in annotations_by_image.items():
        filename = id_to_filename[image_id]
        
        # Extract image dimensions
        image_info = next(filter(lambda x: x['id'] == image_id, annotations['images']))
        height, width = image_info['height'], image_info['width']

        # Create masks for each category
        for annotation in image_annotations:
            category_id = annotation['category_id']

            # Skip if category_id is not in the list
            if category_ids is not None and category_id not in category_ids:
                continue

            # Create an empty mask for the category
            mask = np.zeros((height, width), dtype=np.uint8)

            for segment in annotation['segmentation']:
                # Convert segmentation points to a format suitable for cv2.fillPoly
                segment = np.array(segment, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [segment], 255)

            # Save the mask as TIFF with category_id in filename
            mask_image = Image.fromarray(mask)
            mask_image.save(f"{output_dir}/{filename.split('.')[0]}_cat{category_id}_mask.{image_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, required=True, \
                        help="Path to the COCO JSON file")
    parser.add_argument("--save_dir", type=str, required=True, \
                        help="Directory to save the mask results")
    parser.add_argument("--category_ids", nargs='+', type=int, required=False, \
                        help="List of category ids to extract")
    parser.add_argument("--image_format", type=str, required=False, \
                        help="Image format to save the masks")
    args = parser.parse_args()
    coco_to_mask(args.coco_path, args.save_dir, args.category_ids, args.image_format)