# Slice COCO annotations and images using sahi
# see: https://github.com/obss/sahi

import os
from sahi.slicing import slice_image, slice_coco

def split_image(fname_image, tile_size = 512, output_dir = None, output_file_name = None, out_ext, overlap_ratio = 0.5):
    """
    Slice image into tiles of size tile_size x tile_size pixels.
    Simple wrapper around sahi.slicing.slice_image.

    Args:
        fname_image (str): The filename of the image to slice.
        tile_size (int): The size of the tiles.
        output_dir (str): The directory to save the tiles.
        output_file_name (str): The filename of the output tiles.
        out_ext (str): The extension of the output tiles. Default is the
        original suffix.
        overlap_ratio (float): The overlap ratio between tiles.
    """

    #slice single image
    slice_image_result, num_total_invalid_segmentation = slice_image(
        image=fname_image,
        output_file_name=output_file_name,
        output_dir=output_dir,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        out_ext=out_ext,
    )
    return slice_image_result, num_total_invalid_segmentation

def split_coco(fname_coco, tile_size = 512, image_dir = None, outpath = None, overlap_ratio = 0.5):
    """
    Slice COCO dataset into tiles of size tile_size x tile_size pixels.
    Simple wrapper around sahi.slicing.slice_coco.
    """
    if outpath is None:
        outpath = os.path.dirname(fname_coco)
    else:
        os.makedirs(outpath, exist_ok=True)

    if image_dir is None:
        image_dir = os.path.dirname(fname_coco)

    #slice COCO dataset
    output_coco_fname = os.path.join(outpath, f'coco_tiled_{tile_size}pix.json')
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=fname_coco,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_coco_fname,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio = 0.01,
        verbose=True,
    )

    return coco_dict, coco_path