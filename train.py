import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import sys
import yaml
from os import chdir
from os.path import isfile, join, isdir
from kneedeep.data import MOSTRadio
from kneedeep.models import KneeLocalizer
from kneedeep.io.paths import makedir_if_not_exists
from evaluate import evaluate
from shutil import copy

dataset_classes = {'MOST': MOSTRadio}


def train(conf_path):

    # Load the config file
    with open(conf_path, 'r') as c:
        config_dict = yaml.load(c)
        d = config_dict['dataset']
        out_dir = config_dict['output_dir']

    copy(conf_path, out_dir)

    # Load data
    h5_file = join(config_dict['output_dir'], config_dict['save_name']) + '.h5'
    if isdir(config_dict['data_root']):
        dataset = dataset_classes[d](config_dict['data_root'], out_dir, h5_file)
        dataset.split()
        dataset.generate_hdf5_file(config_dict)

    # Instantiate and train model
    model = KneeLocalizer(config_dict)
    model.train(h5_file)

    # Evaluate model
    eval_dir = join(out_dir, 'evaluate')
    makedir_if_not_exists(eval_dir)
    evaluate(config_dict, 'final_model.h5', config_dict['save_name'] + '.h5')


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    args = parser.parse_args()
    train(args.config)
