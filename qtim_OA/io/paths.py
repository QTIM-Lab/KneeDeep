from os import makedirs
from os.path import split, isdir


def makedir_if_not_exists(dir):

    if not isdir(dir):
        makedirs(dir)


def most_metadata(img_path):

    # Given a MOST path, return the metadata associated with the image
    parts = split(img_path)[-4:]
    keys = ['subjectID', 'visit', 'img_name', 'view']
    return dict(zip(keys, parts))
