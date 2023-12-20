import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set the folder containing the groud-truths, predictions, and raw images to make some plots an estimate IOU score.
gt_folder = "TESTING/masks/"
pred_folder = "TESTING/predictions/"
image_folder = "TESTING/images/"

def binarize_image(image, invert=False):
    if invert:
        _, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY_INV)
    else:
        _, binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)
    return binary_image

def parse_prediction_number(file_name):
    # Extract the prediction number from the file name
    fname=int(file_name.split("_")[1].split(".")[0])
    print(fname)
    return fname

def calculate_iou(gt_binary, pred_binary):
    intersection = np.logical_and(gt_binary, pred_binary)
    union = np.logical_or(gt_binary, pred_binary)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def plot_diagnostic(gt_binary, pred_binary, iou_score, image_file, fname=None):
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    ll=ax.imshow(image_file, cmap='Greys', alpha=1)

    print(np.min(gt_binary), np.max(gt_binary), np.mean(gt_binary), np.median(gt_binary))
    gt_binary[gt_binary<1] = 0
    gt_binary[gt_binary>1] = 1
    # gt_binary = np.ma.masked_where(gt_binary < 0.5, gt_binary)
    # ll1=ax.imshow(gt_binary, cmap='winter', alpha=0.5, vmin=0, vmax=1)
    ll1 = ax.contour(gt_binary, levels=1, colors='yellow', linewidths=1)

    # Plot Prediction in red with opacity
    print(np.min(pred_binary), np.max(pred_binary), np.mean(pred_binary), np.median(pred_binary))
    pred_binary = np.ma.masked_where(pred_binary > 183, pred_binary)
    pred_binary[pred_binary==1] = 1
    pred_binary[pred_binary<1] = 0
    # pred_binary = np.ma.masked_where(pred_binary < 0.5, pred_binary)
    # ll2=ax.imshow(pred_binary, cmap='cool', alpha=0.5,vmin=0, vmax=1)
    ll2 = ax.contour(pred_binary, levels=1, colors='blue', linewidths=1)


    # Plot Overlap in green with opacity
    overlap = np.logical_and(gt_binary, pred_binary)
    print(np.min(overlap), np.max(overlap), np.mean(overlap), np.median(overlap))
    overlap = np.ma.masked_where(overlap < 0.5, overlap)
    # overlap[overlap<1] = 0
    # overlap[overlap>1] = 1
    # ll3=ax.imshow(overlap, cmap='autumn', alpha=0.5, vmin=0, vmax=1)
    ll3 = ax.contourf(overlap, levels=1, colors='red', linewidths=1)
    #plt.colorbar(ll3)

    # Create legend handles and labels
    legend_handles = [
        mpatches.Patch(color='yellow', label='Ground Truth'),
        mpatches.Patch(color='blue', label='Predictions'),
        mpatches.Patch(color='red', label='Overlap')
    ]

    # Add legend to the plot
    ax.legend(handles=legend_handles, loc='lower right')

    l3=ax.set_title('Overlap (IoU={:.2f})'.format(iou_score))
    if fname:
        plt.savefig(fname[:-4]+'_prediction.png')
    plt.show()

def plot_diagnostic_sep(gt_binary, pred_binary, iou_score):
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Ground Truth in blue
    axes[0].imshow(gt_binary, cmap='Blues', interpolation='none')
    axes[0].set_title('Ground Truth')

    # Prediction in red
    axes[1].imshow(pred_binary, cmap='Reds', interpolation='none')
    axes[1].set_title('Prediction')

    # Overlap in green
    overlap = np.logical_and(gt_binary, pred_binary)
    axes[2].imshow(overlap, cmap='Greens', interpolation='none')
    axes[2].set_title('Overlap (IoU={:.2f})'.format(iou_score))

    plt.show()

def calculate_iou_for_folder(gt_folder, pred_folder, image_folder):
    print("GT:",gt_folder)
    print("PRED:",pred_folder)
    iou_scores = []
    files_gt = list(sorted([f for f in os.listdir(gt_folder)]))
    files_pred = list(sorted([f for f in os.listdir(pred_folder)]))
    files_images = list(sorted([f for f in os.listdir(image_folder)]))

    for gt_file, pred_file, image_file in zip(files_gt, files_pred, files_images):
        print(gt_folder+gt_file, pred_folder+pred_file)
        gt_image = cv2.imread(gt_folder+gt_file, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_folder+pred_file, cv2.IMREAD_GRAYSCALE)
        image_actual = cv2.imread(image_folder+image_file, cv2.IMREAD_GRAYSCALE)
        gt_binary = binarize_image(gt_image)
        pred_binary = binarize_image(pred_image, invert=False)
        # plot_diagnostic(gt_image, pred_image, 0)
        iou_score = calculate_iou(gt_binary, pred_binary)
        print(iou_score, gt_file,  pred_file)
        iou_scores.append(iou_score)
        plot_diagnostic(gt_binary, pred_binary, iou_score, image_actual, fname=image_file)
        # if pred_number == 171:
        # plot_diagnostic_sep(gt_binary,pred_binary,iou_score)

    average_iou = np.mean(iou_scores)
    print(min(iou_scores), max(iou_scores))
    return average_iou

iou_score = calculate_iou_for_folder(gt_folder, pred_folder, image_folder)
print("Average IoU Score:", iou_score)
