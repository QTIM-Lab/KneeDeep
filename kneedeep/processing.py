from skimage.exposure import equalize_adapthist
from skimage.morphology import label
from skimage.measure import regionprops


preprocessors = {'clahe': equalize_adapthist}


def normalize(img, method='clahe'):

    func = preprocessors[method]
    return func(img)


def prediction_to_bounding_boxes(pred, thresh=0.5):

    # Threshold and label
    label_image = label(pred > thresh)

    props = regionprops(label_image)
    