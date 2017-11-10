from kneedeep.models import KneeLocalizer
from kneedeep.io.paths import makedir_if_not_exists
from os.path import join, isfile, isdir, splitext
import yaml
import pandas as pd
from glob import glob


def localize_knees(images, out_dir, config_dict, mode='crop'):

    if isfile(images) and splitext(images)[1] == '.csv':
        df = pd.DataFrame.from_csv(images)
        img_list = list(df.index)
    elif isdir(images):
        img_list = glob(join(images, '*'))
    else:
        raise IOError("Unable to load data from '{}'".format(images))

    # Instantiate localizer CNN
    localizer = KneeLocalizer(config_dict, pretrained_model=join(config_dict['output_dir'], 'final_model.h5'))
    makedir_if_not_exists(out_dir)
    initial = mode.lower()[0]

    if initial == 'p':
        localizer.predict(img_list, resize_output=True, save_dir=out_dir)
    elif initial == 'b':
        localizer.get_bounding_boxes(img_list, no_boxes=2, save_dir=out_dir)
    elif initial == 'c':
        localizer.crop(img_list, no_boxes=2, save_dir=out_dir)
    else:
        raise ValueError("Invalid mode ''. Choose from 'predict', 'bbox', and 'crop'.".format(mode))


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--images', dest='images', required=True)
    parser.add_argument('-o', '--out-dir', dest='out_dir', required=True)
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-m', '--mode', dest='mode', type=str, default='crop')
    args = parser.parse_args()

    with open(args.config, 'rb') as f:
        config = yaml.load(f)

    localize_knees(args.images, args.out_dir, config, mode=args.mode)
