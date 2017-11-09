from kneedeep.models import KneeLocalizer
from kneedeep.io.paths import makedir_if_not_exists
from os.path import join, isfile, splitext
import yaml
import pandas as pd
import numpy as np


def localize_knees(config, images):

    if isfile(images) and splitext(images)[1] == '.csv':
        df = pd.DataFrame.from_csv(images)
        img_list = list(df.index)
    else:
        raise IOError("Unable to load data from '{}'".format(images))

    inf_dir = join(config['output_dir'], 'inference')
    makedir_if_not_exists(inf_dir)

    # Instantiate localizer CNN
    joint_localizer = KneeLocalizer(config, pretrained_model=join(config['output_dir'], 'final_model.h5'))
    joint_localizer.predict(img_list, save_dir=inf_dir)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-i', '--images', dest='images', required=True)
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = yaml.load(f)

    localize_knees(config, args.images)
