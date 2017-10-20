import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import sys
import yaml
from os.path import isfile, join, isdir
from qtim_OA.data import MOSTRadio
from qtim_OA.models import JointSegmenter

dataset_classes = {'MOST': MOSTRadio}


def train(conf_path):

    # Load the config file
    with open(conf_path, 'r') as c:
        config_dict = yaml.load(c)
        d = config_dict['dataset']
        m = config_dict['model']
        out_dir = config_dict['output_dir']

    # Load data
    h5_file = join(config_dict['output_dir'], config_dict['save_name']) + '.h5'
    if isdir(config_dict['data_root']):
        dataset = dataset_classes[d](config_dict['data_root'], out_dir, h5_file)
        dataset.split()
        dataset.generate_hdf5_file(config_dict)

    # Instantiate model
    model = JointSegmenter(config_dict)
    model.train(h5_file)

if __name__ == '__main__':

    train(sys.argv[1])
