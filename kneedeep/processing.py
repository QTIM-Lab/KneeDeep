from skimage.exposure import equalize_adapthist
from skimage.morphology import label
from skimage.measure import regionprops
from scipy.misc import imresize
import numpy as np


def apply_to_batch(inputs, func, **params):

    output = []
    for input_data in zip(inputs):
        result = func(*input_data, **params)
        output.append(result)
    return np.asarray(output)


def resize(img, new_size, mode='bilinear'):
    return imresize(img, new_size, mode=mode)


def normalize(img, method='clahe'):

    func = preprocessors[method]
    return func(img)


def prediction_to_bounding_boxes(pred, thresh=0.5):

    # Threshold and label
    label_image = label(pred > thresh)
    props = sorted(regionprops(label_image), key=lambda x: x.area, reverse=True)
    return [region.bbox for region in props[:2]]


def crop_to_bounding_box(img, bboxes):

    cropped = []
    for bbox in bboxes:
        img_crop = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]  # min_row, min_col, max_row, max_col
        cropped.append(img_crop)
    return cropped


preprocessors = {'clahe': equalize_adapthist}
postprocessors = {'resize': resize, 'bbox': prediction_to_bounding_boxes, 'crop': crop_to_bounding_box}
