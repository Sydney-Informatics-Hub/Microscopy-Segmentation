'''
Script for converting anylabeling JSON format to YOLO format
and splitting it into train and validation sets.

Forked from: https://github.com/ThijsCol/Anylabeling-LabelMe-json-to-yolo-txt

Usage:
    python anylableing2yolo.py --class_labels '{"label1": 0, "label2": 1}' --input_dir 'input_data' --output_dir 'output_data' --split_ratio 0.2

Arguments:
    --class_labels (dict): Class label dictionary mapping label names to numeric class indices.
                           Defaults to {"fat": 0, "mit": 1}.
    --input_dir (str): Input directory containing images and corresponding JSON annotation files.
                       Defaults to './anylabeling_data'.
    --output_dir (str): Output directory where YOLO-formatted data will be saved.
                        Defaults to './data_yolo'.
    --split_ratio (float): Train-validation split ratio. If set to 0, all data will be used for training.
                           Defaults to 0.2 (20% validation).

Note:
    The script assumes that each JSON file corresponds to an image file and that
    the image file is present in the input directory with extensions '.tif', '.png', or '.jpeg'.
    The YOLO-formatted annotations will be saved in separate directories for train and validation sets.

Output:
    The script generates YOLO-formatted annotation files in the specified output directory
    for both the training and validation sets. It copies the image files and creates a corresponding
    .txt file with the YOLO annotations.

Example:
    python script.py --class_labels '{"person": 0, "car": 1}' --input_dir './image_data' --output_dir './datasets' --split_ratio 0.1
'''

import os
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Your script description here.')

# Add command line arguments
parser.add_argument('--class_labels', type=dict, default={"fat": 0, "mit": 1}, help='Class label dictionary. Extend  as needed.')
parser.add_argument('--input_dir', type=str, default='./anylabeling_data', help='Input directory containing images and json files')
parser.add_argument('--output_dir', type=str, default='./datasets', help='Output directory')
parser.add_argument('--split_ratio', type=float, default=0.2, help='Train-validation split ratio')

args = parser.parse_args()

class_labels = args.class_labels
input_dir = args.input_dir
output_dir = args.output_dir
split_ratio = args.split_ratio

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create train and validate directories
train_dir = os.path.join(output_dir, 'images/train')
train_dir_labels = os.path.join(output_dir, 'labels/train')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_dir_labels, exist_ok=True)

validate_dir = os.path.join(output_dir, 'images/validate')
validate_dir_labels = os.path.join(output_dir, 'labels/validate')
if split_ratio > 0:
    os.makedirs(validate_dir, exist_ok=True)
    os.makedirs(validate_dir_labels, exist_ok=True)

json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.tif', '.png', '.jpeg'))]

if split_ratio > 0:
    train_images, validate_images = train_test_split(image_files, test_size=split_ratio)
else:
    train_images = image_files

# Copy all images to train and validate directories
print("Copying images")
for image_file in tqdm(image_files, desc="Copying images"):
    current_output_dir = train_dir if image_file in train_images else validate_dir
    shutil.copy(os.path.join(input_dir, image_file), current_output_dir)

# Use tqdm for progress bar
for filename in tqdm(json_files, desc="Copying annotations"):
    print(filename)
    with open(os.path.join(input_dir, filename)) as f:
        data = json.load(f)

    image_filename = filename.replace('.json', '')
    if any(os.path.isfile(os.path.join(input_dir, image_filename + ext)) for ext in ['.tif', '.png', '.jpeg']):
        if image_filename + '.tif' in train_images or image_filename + '.png' in train_images or image_filename + '.jpeg' in train_images:
            current_output_dir = train_dir_labels
        else:
            current_output_dir = validate_dir_labels

        with open(os.path.join(current_output_dir, filename.replace('.json', '.txt')), 'w') as out_file:
            for shape in data['shapes']:
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]

                dw = 1./data['imageWidth']
                dh = 1./data['imageHeight']
                w = x2 - x1
                h = y2 - y1
                x = x1 + (w/2)
                y = y1 + (h/2)

                x *= dw
                w *= dw
                y *= dh
                h *= dh

                class_label = shape['label']
                # print(class_label, shape, print, class_labels)
                out_file.write(f"{class_labels[class_label]} {x} {y} {w} {h}\n")

print("Conversion and split completed successfully!")
