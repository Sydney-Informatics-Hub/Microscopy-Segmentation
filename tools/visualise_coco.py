"""
Visualize COCO annotations on an image.
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

def visualize_coco_segmentation(infile_img, infile_json, outfname=None):
    """
    This function visualizes an image with the COCO segmentation polygons overlaid.

    Args:
        infile_img (str): The filename of the image to visualize.
        infile_json (str): The filename of the COCO JSON file.
        outfname (str): Output filename. If None, the image is not saved.
    """
    # Load the COCO JSON file
    with open(infile_json) as f:
        data = json.load(f)
    
    # Read the image
    img = cv2.imread(infile_img)

    # get filename without path
    infile_img = os.path.basename(infile_img)

    # search i data['images'] for the image with the given id
    for image in data['images']:
        if image['file_name'] == infile_img:
            # get image id
            img_id = image['id']
            break
    if img_id is None:
        raise ValueError('Image not found in JSON file')
    
    # For each annotation in the COCO JSON file
    for annotation in data['annotations']:
        # If the annotation is for the given image
        if annotation['image_id'] == img_id:
            # For each segmentation in the annotation
            for segmentation in annotation['segmentation']:
                # Convert the segmentation to a numpy array and reshape it to a 2D array
                polygon = np.array(segmentation).reshape(-1, 2).astype(np.int32)
                
                # Draw the polygon on the image
                cv2.polylines(img, [polygon], True, (255, 0, 0), 2)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    if outfname is not None:
        plt.savefig(outfname, bbox_inches='tight')
    plt.show()

def visualize_coco_detection(infile_img, infile_coco, outfname=None):
    """
    This function visualizes an image with the COCO bounding boxes and polygon segmentation overlaid.
    This assumes the dataset includes bounding box annotations.

    Args:
        infile_img (str): Input image file.
        infile_coco (str): Input COCO JSON file.
        outfname (str): Output filename. If None, the image is not saved.
    """
    cocodata = json.load(open(infile_coco, 'r'))
    img = cv2.imread(infile_img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=cocodata, scale=0.5)
    out = visualizer.draw_dataset_dict(cocodata)
    plt.figure(figsize=(10, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    if outfname is not None:
        plt.savefig(outfname, bbox_inches='tight')
    plt.show()
