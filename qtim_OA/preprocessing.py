from skimage.exposure import equalize_adapthist

preprocessors = {'clahe': equalize_adapthist}


def normalize(img, method='clahe'):

    func = preprocessors[method]
    return func(img)
