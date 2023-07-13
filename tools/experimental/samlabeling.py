"""
This us an experimental script to generate masks for an image using the SAM model.

Outline for app:
1. User uploads image
2. User selects points on image or draws a bounding box
3. Mask is generated, displayed, and saved as json and tif

The SAM model can be loaded with 3 different encoders: ViT-B, ViT-L, and ViT-H. 
ViT-H improves substantially over ViT-B but has only marginal gains over ViT-L. 
These encoders have different parameter counts, with ViT-B having 91M, 
ViT-L having 308M, and ViT-H having 636M parameters. 
This difference in size also influences the speed of inference, 
so keep that in mind when choosing the encoder for your specific use case.
"""


from IPython.display import display, HTML
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
from skimage import measure
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# local imports
from imagepointer import PointAndBoxSelector

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_image_orig(infname):
    image = cv2.imread(infname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_points_on_image(infname, points):
    image = cv2.imread(infname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(7,7))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show() 

def show_image_annot(infname, masks):
    # plot two subfigures of original image next to image with annotations
    image = cv2.imread(infname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.imshow(image)
    # give image a title
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(image)
    show_anns(masks)
    plt.title('Image with Annotations')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_image_masks_loop(image, masks):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()  

def show_3masks(image, masks, scores):
    plt.figure(figsize=(15,6))

    plt.subplot(1,3,1)
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask 1, Score: {scores[0]:.3f}", fontsize=14)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(image)
    show_mask(masks[1], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask 2, Score: {scores[1]:.3f}", fontsize=14)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(image)
    show_mask(masks[2], plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask 3, Score: {scores[2]:.3f}", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()  


def sam_generate(infname, outpath, save_tif=False, preview=False):
    """
    Interactively generate polygons for an image using the SAM model.
    Polygons results are saved as json files in the specified output path.
    Optionally the polygons are also saved as tif images.

    Parameters
    ----------
    infname : str
        Path + filename for input image.
    outpath : str
        Path to output folder.
    save_tif : bool
        If True, polygon masks are also saved as tif images.
    preview : bool
        If True, preview of polygons is shown.
    """
    # define SAM model
    # checkpoints: https://github.com/facebookresearch/segment-anything#model-checkpoints
    sam_checkpoint = "checkpoints_sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    #device = torch.device("mps") # doesn't work since metal doesn't support float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    mask_generator_custom = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    # generate masks for an image
    show_image_orig(infname)

    masks = mask_generator.generate(image)

    print('Mask length:', len(masks))
    print('Mask keys:', masks[0].keys())
    show_image_annot(infname, masks)

    # Interactively select point from image and get coordinates of points
    pbs = PointAndBoxSelector()
    pbs.select_points_and_boxes(infname)
    cv2.destroyAllWindows()
    input_points = pbs.points

    predictor = SamPredictor(sam)

    image = cv2.imread(infname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    for i, input_point in enumerate(input_points):
        print(f"Extracting polygon for point {i+1}: {input_point}")

        input_point = np.asarray([input_point])
        input_label = np.array([i])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_best = masks[0] 

        #print('Number of masks found:', masks.shape[0])
        if preview:
            show_3masks(image, masks, scores)
           
        # save mask array to image tif
        mask_best = mask_best.astype(np.uint8)
        mask_best = mask_best * 255
        if save_tif:
            cv2.imwrite('mask_best.tif', mask_best)

        # convert mask to polygon
        contours = measure.find_contours(mask_best, 0.5, fully_connected='high')
        print('Number of contours found:', len(contours))
        # save polygon to json
        with open(f'contours_{i}.json', 'w') as f:
            json.dump(contours, f)