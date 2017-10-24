import keras.backend as K
from keras.layers import Input, Activation, UpSampling2D, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Nadam
from os.path import join
import tables
import numpy as np


class JointSegmenter:

    def __init__(self, config, arch='unet'):

        if config['backend'] == 'theano':
            input_shape = (1, config['resize']['height'], config['resize']['width'])
            order = 'channels_first'
        else:
            input_shape = (config['resize']['height'], config['resize']['width'], 1)
            order = 'channels_last'

        self.epochs = config['epochs']
        self.model = architectures[arch](input_shape=input_shape, order=order)
        self.out_dir = config['output_dir']

    def train(self, h5_file):

        data = tables.open_file(h5_file, mode="r").root.train
        img_data = data.images
        label_data = data.labels

        self.model.fit(np.asarray(img_data), np.asarray(label_data), epochs=self.epochs, batch_size=16)
        self.model.save(join(self.out_dir, 'final_model.h5'))


class JointClassifier:

    def __init__(self):
        pass


def joint_unet(input_shape=(480, 576, 1), filter_divisor=1, pool_size=(2, 2), activation='relu', order='channels_last'):

    concat_axis = -1 if order == 'channels_last' else 1

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(int(32/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(inputs)
    conv1 = Conv2D(int(64/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size, data_format=order,)(conv1)

    conv2 = Conv2D(int(64/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(pool1)
    conv2 = Conv2D(int(128/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size, data_format=order)(conv2)

    conv3 = Conv2D(int(128/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(pool2)
    conv3 = Conv2D(int(256/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size, data_format=order)(conv3)

    conv4 = Conv2D(int(256/filter_divisor), (3, 3), activation=activation,data_format=order,
                   padding='same')(pool3)
    conv4 = Conv2D(int(512/filter_divisor), (3, 3), activation=activation,data_format=order,
                   padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)

    up5 = UpSampling2D(data_format=order)(conv4)
    up5 = concatenate([up5, conv3], axis=concat_axis)

    conv5 = Conv2D(int(256/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(up5)
    conv5 = Conv2D(int(256/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(data_format=order)(conv5)
    up6 = concatenate([up6, conv2], axis=concat_axis)

    conv6 = Conv2D(int(128/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(up6)
    conv6 = Conv2D(int(128/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(data_format=order)(conv6)
    up7 = concatenate([up7, conv1], axis=concat_axis)

    conv7 = Conv2D(int(64/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(up7)
    conv7 = Conv2D(int(64/filter_divisor), (3, 3), activation=activation, data_format=order,
                   padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    conv8 = Conv2D(1, (1, 1), data_format=order)(conv7)

    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)
    model.compile(optimizer=Nadam(), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

architectures = {'unet': joint_unet}
