from skimage.exposure import equalize_adapthist
from skimage.morphology import label, binary_dilation, disk
from skimage.measure import regionprops
from skimage.transform import resize
import numpy as np


def apply_to_batch(inputs, func_name, **params):

    output = []
    func = postprocessors[func_name]
    for input_data in zip(*inputs):
        result = func(*input_data, **params)
        output.append(result)
    return output


def normalize(img, method='clahe'):

    func = preprocessors[method]
    return func(img)


def prediction_to_bounding_boxes(pred, no_boxes=2, dilation_factor=.02, thresh=0.5):

    # Threshold and label
    dilate_pixels = max(pred.shape) * dilation_factor
    label_image = label(binary_dilation(pred > thresh, disk(dilate_pixels)))
    props = sorted(regionprops(label_image), key=lambda x: x.area, reverse=True)

    no_boxes = min(len(props), no_boxes)
    return [list(region.bbox) for region in sorted(props[:no_boxes], key=lambda x: x.bbox[0])]


def crop_to_bounding_box(img, bboxes):

    cropped = []
    for min_row, min_col, max_row, max_col in bboxes:
        img_crop = img[min_row:max_row, min_col:max_col]
        cropped.append(img_crop)
    return cropped


preprocessors = {'clahe': equalize_adapthist}
postprocessors = {'resize': resize, 'bbox': prediction_to_bounding_boxes, 'crop': crop_to_bounding_box}
