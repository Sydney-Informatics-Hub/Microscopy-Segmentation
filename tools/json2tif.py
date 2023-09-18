import json
from PIL import Image, ImageDraw
import numpy as np

# Load the JSON data
with open('HM25_HighRes_Aligned0020.json', 'r') as json_file:
    data = json.load(json_file)

# Create an empty image with the specified dimensions
image = Image.new('L', (data["imageWidth"], data["imageHeight"]), 0)

# Create a draw object to draw polygons on the image
draw = ImageDraw.Draw(image)

# Iterate over the shapes and collect all unique labels
all_labels=[]
for shape in data["shapes"]:
    label = shape["label"]
    if label not in all_labels:
        all_labels.append(label)

# Create a mapping of labels to unique pixel values
label_to_pixel = {label: i+1 for i, label in enumerate(all_labels)}

print(label_to_pixel)

# Iterate over the shapes and draw the polygons with unique pixel values
for shape in data["shapes"]:
    label = shape["label"]
    points = [(int(p[0]), int(p[1])) for p in shape["points"]]    
    pixel_value = label_to_pixel.get(label, 0)  # Default to 0 if label is not in the mapping
    draw.polygon(points, fill=pixel_value)

# Save the image as a TIFF file
image.save('output4.tiff')

# Close the image
image.close()