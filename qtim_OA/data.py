from os.path import join, split, splitext, isfile
from glob import glob
import pandas as pd
from pathlib import PurePath
from random import shuffle
import dicom
from scipy.misc import imread, imresize
import tables
import numpy as np
from .preprocessing import normalize


class Dataset(object):

    def __init__(self, data_root, output_dir):

        self.data_root = data_root
        self.output_dir = output_dir

    def get_meta_data(self, img_path):
        return NotImplementedError()


class MOSTRadio(Dataset):

    def __init__(self, data_root, output_dir, out_name):

        super(MOSTRadio, self).__init__(data_root, output_dir)
        self.search_pattern = join(data_root, 'XR', '*', '*', '*', 'PA*', '*')
        self.keys = ['visit', 'subjectID', 'view', 'image_name']

        self.img_paths = glob(self.search_pattern)
        self.label_ext = '_mask.tif'
        meta_data = [self.get_meta_data(f) for f in self.img_paths if not splitext(f)[1] and isfile(f + self.label_ext)]

        self.random_state = 101
        self.meta_data = pd.DataFrame(data=meta_data).set_index('img_path')
        self.train, self.validation, self.test = None, None, None

        self.out_name = out_name

    def get_meta_data(self, img_path):

        parts = PurePath(img_path).parts[-4:]
        meta_dict = dict(zip(self.keys, parts))
        meta_dict['img_path'] = img_path
        return meta_dict

    def split(self, train=.7, validation=.1, test=.2, on='subjectID'):

        train_out = join(self.output_dir, 'train.csv')
        val_out = join(self.output_dir, 'validation.csv')
        test_out = join(self.output_dir, 'test.csv')

        if isfile(train_out) and isfile(val_out) and isfile(test_out):
            self.train = pd.DataFrame.from_csv(train_out)
            self.validation = pd.DataFrame.from_csv(val_out)
            self.test = pd.DataFrame.from_csv(test_out)
            return

        split_sum = sum([x for x in [train, validation, test] if x is not None])

        if not train or round(split_sum) != 1.:
            raise ValueError("Invalid split sizes - must sum to 1")

        unique_entries = self.meta_data[on].unique()
        shuffle(unique_entries)

        train_samples = int(len(unique_entries) * train)
        val_samples = int(len(unique_entries) * validation)
        test_samples = int(len(unique_entries) * test)

        train_indices = unique_entries[:train_samples]
        val_indices = unique_entries[train_samples:train_samples+val_samples]
        test_indices = unique_entries[train_samples+val_samples:train_samples+val_samples+test_samples]

        self.train = self.meta_data[self.meta_data[on].isin(train_indices)]
        self.validation = self.meta_data[self.meta_data[on].isin(val_indices)]
        self.test = self.meta_data[self.meta_data[on].isin(test_indices)]

        self.train.to_csv(train_out)
        self.validation.to_csv(val_out)
        self.test.to_csv(test_out)

    def generate_hdf5_file(self, config):

        # Open HDF5 file
        if isfile(out_name) and not config['overwrite_data']:
            return out_name

        h5file = tables.open_file(out_name, mode="w", title=out_name)
        img_dtype = tables.Float16Atom()
        print("Generating HDF5 file")

        new_size = (config['resize']['height'], config['resize']['width'])
        img_shape = (0,) + new_size + (1,)

        for prefix, df in {'train': self.train, 'val': self.validation, 'test': self.test}.items():

            group = h5file.create_group("/", prefix)
            img_storage = h5file.create_earray(group, 'images', img_dtype, shape=img_shape)
            label_storage = h5file.create_earray(group, 'labels', img_dtype, shape=img_shape)
            filename_storage = h5file.create_earray(group, 'filenames', tables.StringAtom(256), shape=(0,))

            for img_path, meta_data in df.iterrows():

                print(img_path)
                filename_storage.append([img_path])

                # Load image and labels
                img_arr = dicom.read_file(img_path, force=True).pixel_array
                label_arr = imread(img_path + self.label_ext)

                img_arr_resized = imresize(img_arr, new_size, interp='bilinear')
                label_arr_resized = imresize(label_arr, new_size, interp='nearest')

                img_arr_pp = normalize(img_arr_resized).astype(np.float16)
                label_arr_pp = (label_arr_resized / 255.).astype(np.float16)

                img_arr_pp = np.expand_dims(img_arr_pp, 2)
                label_arr_pp = np.expand_dims(label_arr_pp, 2)

                img_storage.append(img_arr_pp[None])  # this syntax prepends a singlton dimension to the image
                label_storage.append(label_arr_pp[None])

        h5file.close()
        return out_name
