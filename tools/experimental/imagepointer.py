"""
Experimental script to interactively select points and bounding boxes in an image.
CURRENTLY NOT WORKING FOR POINTS, ONLY FOR BOUNDING BOXES.

Example
-------
from imagepointer import PointAndBoxSelector as pbs
pbs.select_points_and_boxes('image.png')
"""

import cv2
import numpy as np
import os


class PointAndBoxSelector:

    def __init__(self):
        # Initialize the list of points and boolean indicating
        # whether cropping is being performed or not
        self.points = []
        self.rect_endpoint_tmp = []
        self.rectangles = []
        self.drawing = False

    def select_points_and_boxes(self, img_path):
        # Load the image, clone it, and setup the mouse callback function
        self.image = cv2.imread(img_path)
        self.clone = self.image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)
        print('Instructions for selection:')
        print('-----------------------------------')
        print('Select points by clicking on image.')
        print('Select boxes by click and drag.')
        print('Press "r" to reset selection.')
        print('Press "c" to complete.')

        # Keep looping until the 'q' key is pressed
        while True:
            # Display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            # If the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = self.clone.copy()
                self.points = []
                self.rectangles = []

            # If the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                cv2.destroyAllWindows()
                break

        print(f"Points Selected: {self.points}")
        print(f"Bounding Boxes Selected: {self.rectangles}")

    def click_and_crop(self, event, x, y, flags, param):
        # If the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.rect_endpoint_tmp = [(x, y)]
            self.drawing = True

        # Check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # Record the ending (x, y) coordinates
            self.rect_endpoint_tmp.append((x, y))
            self.drawing = False
            self.rectangles.append(tuple(self.rect_endpoint_tmp))
            # Draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.rect_endpoint_tmp[0], self.rect_endpoint_tmp[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)

        # does not work for points, need EVENT_MOUSEMOVE
        # see https://stackoverflow.com/questions/22140880/drawing-rectangle-or-line-using-mouse-events-in-open-cv-using-python



def test_PointAndBoxSelector():
    import requests
    # make temporary directory and download images
    os.makedirs(name='images', exist_ok=True)
    # download example images and save to images folder

    # Send a HTTP request to the URL of the image
    url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    response = requests.get(url)

    file_path = 'images/test.jpg'

    # Check the HTTP response status
    if response.status_code == 200:
        # Open the output file in binary mode, write the response content to this file
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Failed to download the image, HTTP response status code {response.status_code}")

    # initialize the PointAndBoxSelector and select points and boxes
    pbs = PointAndBoxSelector()

    # Pass image to our PointAndBoxSelector
    pbs.select_points_and_boxes(file_path)

    cv2.destroyAllWindows()