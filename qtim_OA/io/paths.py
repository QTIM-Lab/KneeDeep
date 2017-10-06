from os.path import split



def most_metadata(img_path):

    # Given a MOST path, return the metadata associated with the image
    parts = split(img_path)[-4:]
    keys = ['subjectID', 'visit', 'img_name', 'view']
    return dict(zip(keys, parts))
