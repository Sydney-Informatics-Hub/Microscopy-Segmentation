import os
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from scipy import ndimage

# Set the shape of each image in the stack
img_shape = (4096, 4096)

# Specify the directory containing TIFF images
#tiff_dir = '../MITO-EM/OUTPUT/EM30-H-mito-train/PREDICTIONS_val'
tiff_dir = '../MITO-EM/OUTPUT/EM30-R-mito-train-val-v2/PREDICTIONS_val'

# Specify the name of the output HDF5 file
#h5file_name = '0_human_instance_seg_pred.h5'
h5file_name = '1_rat_instance_seg_pred.h5'

# Get a list of TIFF files in the directory
tiff_ids = sorted(next(os.walk(tiff_dir))[2])

# Allocate memory for the image stack
img_stack = np.zeros((len(tiff_ids),) + img_shape, dtype=np.int64)

# Read all TIFF images and stack them
for n, id_ in tqdm(enumerate(tiff_ids), desc='Reading TIFFs'):
    img = imread(os.path.join(tiff_dir, id_))
    img_stack[n] = img

# Apply connected components to create instance segmentation
img_stack = (img_stack / 255).astype('int64')
img_stack, nr_objects = ndimage.label(img_stack)
print("Number of objects: {}".format(nr_objects))

# Create the HDF5 file (using LZF compression to save space)
h5f = h5py.File(h5file_name, 'w')
h5f.create_dataset('dataset_1', data=img_stack, compression="lzf")
h5f.close()
