import keras.backend as K
from keras.layers import Input, Activation, UpSampling2D, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Nadam
from keras.callbacks import Callback, ModelCheckpoint
from os.path import join
from .io.paths import makedir_if_not_exists
from .io.image import save_figure
from .metrics import dice, roc_analysis
import tables
import numpy as np
import pandas as pd
from collections import OrderedDict
from PIL import Image
import seaborn as sns


class KneeLocalizer:

    def __init__(self, config, arch='unet', pretrained_model=None):

        if config['backend'] == 'theano':
            input_shape = (1, config['resize']['height'], config['resize']['width'])
            order = 'channels_first'
        else:
            input_shape = (config['resize']['height'], config['resize']['width'], 1)
            order = 'channels_last'

        self.epochs = config['epochs']
        optimizer = config['optimizer']

        if pretrained_model is None:
            self.model = architectures[arch](input_shape=input_shape, order=order, optimizer=optimizer)
        else:
            self.model = load_model(pretrained_model, custom_objects={'dice_coef': dice_coef})

        self.out_dir = config['output_dir']
        self.callback_dir = join(self.out_dir, 'progress')

    def train(self, h5_file):

        data = tables.open_file(h5_file, mode="r").root
        train_data, val_data = data.train, data.val

        makedir_if_not_exists(self.callback_dir)
        checkpoint_cb = ModelCheckpoint(join(self.out_dir, 'weights_epoch{epoch:02d}_loss{val_loss:.2f}.h5'),
                                        save_best_only=True)
        segment_cb = LocalizeKneeCallback(np.asarray(val_data.images), np.asarray(val_data.labels),
                                          self.callback_dir, no_samples=36)

        # TODO CLAHE, verify input + GT correspond, intensity ranges, etc...
        history = self.model.fit(np.asarray(train_data.images), np.asarray(train_data.labels),
                                 validation_data=[np.asarray(val_data.images), np.asarray(val_data.labels)],
                                 epochs=self.epochs, batch_size=16, callbacks=[checkpoint_cb, segment_cb])

        pd.DataFrame(history.history).to_csv(join(self.out_dir, 'history.csv'))
        self.model.save(join(self.out_dir, 'final_model.h5'))

    def fine_tune(self, data, epochs, model_name='fine_tuned'):

        # TODO Facility to fine-tune an existing network
        raise NotImplementedError('Function not yet implemented!')

    def predict(self, data, mode='crop', resize_output=False, save=None):
        """
        Function to
        :param data: a single image path or list of image paths to segment
        :param mode: mode of output prediction. 'pred' provides the raw (sigmoid) output, 'bbox' provides the image with
                     two bounding boxes overlaid, 'crop' returns the cropped regions of the input image
        :param resize_output: if true, the resulting output will be the same size/resolution as the input data
        :param save: option to save output as an image, if a valid output folder is specified
        :return: the output of the CNN according the mode specified
        """

        # TODO Resize the image(s) according to the config to CNN resolution
        # TODO Normalize the images(s) according to the config file (e.g. CLAHE)
        # TODO Get prediction(s)
        # TODO Process the output(s) according to the mode/resize options
        # TODO Save the results to a folder
        # TODO Return the results

        raise NotImplementedError('Function not yet implemented!')

    def evaluate(self, h5_file, eval_dir):
        """
        Function evaluate the performance of a model on the supplied data. This includes training, validation and test
        :param h5_file: HDF5 file of pre-processed images upon which the loaded model is evaluated
        :param eval_dir: directory in which results are to be saved
        :return: two DataFrames; a summary of metrics (sensitivity, specificity, mean dice, AUROC) and per image dice
        """

        dataset = tables.open_file(h5_file, mode="r").root
        data_splits = OrderedDict([('train', dataset.train), ('val', dataset.val), ('test', dataset.test)])
        results = []
        dice_results = []

        for name, data in data_splits.items():

            y_pred = self.model.predict(np.asarray(data.images), batch_size=16)
            y_true = np.asarray(data.labels)

            # Reshape arrays
            new_shape = (y_pred.shape[0], y_pred.shape[1] * y_pred.shape[2])

            y_pred = y_pred.reshape(new_shape)
            y_true = y_true.reshape(new_shape)

            dice_scores = [dice(yt, yp) for yt, yp in zip(y_true, y_pred)]
            for filename, dsc in zip(data.filenames, dice_scores):
                dice_results.append({'filename': filename, 'Data': name, 'Dice': dsc})

            # ROC analysis
            save_path = join(eval_dir, 'roc_analysis') if name == 'test' else None  # only save the test data
            sensitivity, specificity, auroc, youden, threshold = roc_analysis(y_true.flatten(), y_pred.flatten(), save_path=save_path)
            print youden

            results.append({'Dataset': name, 'Mean dice': np.mean(dice_scores),
                            'Sensitivity': sensitivity[youden],
                            'Specificity': specificity[youden],
                            'Threshold': threshold,
                            'Youden J': youden, 'AUROC': auroc})

        # Plot dice scores
        df_dice = pd.DataFrame(data=dice_results).set_index('filename')
        sns.boxplot(data=df_dice, x='Data', y='Dice')
        save_figure(join(eval_dir, 'dice_results'))

        return pd.DataFrame(data=results), df_dice


class KneeClassifier:

    def __init__(self):
        pass


class LocalizeKneeCallback(Callback):

    def __init__(self, sample_data, sample_labels, out_dir, no_samples=10):

        super(LocalizeKneeCallback, self).__init__()
        self.sample_data = sample_data
        self.sample_labels = sample_labels
        self.out_dir = out_dir
        self.no_samples = no_samples

        self.raw_dir = join(self.out_dir, 'raw')
        self.pred_dir = join(self.out_dir, 'prediction')

        makedir_if_not_exists(self.raw_dir)
        makedir_if_not_exists(self.pred_dir)

    def on_train_begin(self, logs=None):

        input_data = self.sample_data[:self.no_samples]
        for i in range(0, input_data.shape[0]):

            arr = input_data[i, :, :, 0]
            arr = ((arr / np.max(arr)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(join(self.raw_dir, '{}.bmp'.format(i)))

    def on_epoch_end(self, epoch, logs=None):

        input_data = self.sample_data[:self.no_samples]
        output = self.model.predict_on_batch(input_data)

        for i in range(0, output.shape[0]):
            makedir_if_not_exists(join(self.pred_dir, str(i)))
            arr = np.round(output[i, ...] * 255).astype(np.uint8)[:, :, 0]
            Image.fromarray(arr).save(join(self.pred_dir, str(i), '{}.bmp'.format(epoch)))


def joint_unet(input_shape=(480, 576, 1), filter_divisor=1, pool_size=(2, 2), activation='relu', order='channels_last',
               optimizer='nadam'):

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
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[dice_coef])

    return model


def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):

    return 1 - dice_coef(y_true, y_pred)


architectures = {'unet': joint_unet}
