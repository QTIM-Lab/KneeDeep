import matplotlib  # set matplotlib backend
matplotlib.use('Agg')

import sys
import yaml
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
    dataset = dataset_classes[d](config_dict['data_root'], out_dir)
    dataset.split()
    h5_file = dataset.generate_hdf5_file(config_dict)

    # Instantiate model
    model = JointSegmenter(config_dict)
    model.train(h5_file)

if __name__ == '__main__':

    train(sys.argv[1])
