import keras.backend as K
from keras.layers import Input, Activation, UpSampling2D, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import Callback, ModelCheckpoint
from .data import load
from .processing import apply_to_batch
from .io.image import save_bb_overlay
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
from tifffile.tifffile import imsave


class KneeLocalizer:

    def __init__(self, config, arch='unet', pretrained_model=None):

        self.config = config

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

        history = self.model.fit(np.asarray(train_data.images), np.asarray(train_data.labels),
                                 validation_data=[np.asarray(val_data.images), np.asarray(val_data.labels)],
                                 epochs=self.epochs, batch_size=16, callbacks=[checkpoint_cb, segment_cb])

        pd.DataFrame(history.history).to_csv(join(self.out_dir, 'history.csv'))
        self.model.save(join(self.out_dir, 'final_model.h5'))

    def fine_tune(self, data, epochs, model_name='fine_tuned'):

        # TODO Facility to fine-tune an existing network
        raise NotImplementedError('Function not yet implemented!')

    def predict(self, data, thresh=.5, resize_output=True, save_dir=None):
        """
        Performs inference on provided data, and optionally resizes/saves output
        :param data: a single image path or list of image paths to segment
        :param thresh: threshold to apply to predictions when saved as an image
        :param resize_output: option to return results at original resolution
        :param save_dir: folder to save output as .npz
        :return: the raw output of the CNN
        """

        # Pre-process according to config
        raw, preprocessed, filenames, original_shapes = load(data, self.config)

        # Run inference
        print "Running inference..."
        pred = self.model.predict(preprocessed)
        pred = np.squeeze(pred, axis=-1)  # drop redundant channels dimension

        if resize_output:
            pred = apply_to_batch([pred, original_shapes], 'resize')

        if save_dir is not None:
            t = thresh if isinstance(thresh, float) and 0. < thresh < 1. else 0.5
            for img, filename in zip(pred, filenames):
                pil_img = Image.fromarray((img > t).astype(np.uint8) * 255)
                pil_img.save(join(save_dir, filename + '_pred.png'))

        return raw, pred, filenames

    def get_bounding_boxes(self, data, no_boxes=2, save_dir=None):
        """
        Generates bounding boxes from CNN predictions.
        :param data:
        :param no_boxes: maximum number of bounding boxes to find and return
        :param resize_output: option to return results at original resolution
        :param save_dir: folder to save bounding boxes and overlays
        :return: raw CNN predictions and bounding boxes
        """

        # Run inference and locate bounding boxes
        raw, pred, filenames = self.predict(data, resize_output=True, save_dir=save_dir)
        print "Extracting bounding boxes..."
        bboxes = apply_to_batch([pred], 'bbox', no_boxes=no_boxes)

        if save_dir is not None:
            for raw_img, boxes, filename in zip(raw, bboxes, filenames):
                save_bb_overlay(raw_img, boxes, join(save_dir, filename + '_bbox.png'))

        return raw, pred, filenames, bboxes

    def crop(self, data, no_boxes=2, save_dir=None):

        # Extract bounding boxes and crop data
        raw, pred, filenames, bboxes = self.get_bounding_boxes(data, no_boxes=no_boxes, save_dir=save_dir)

        cropped = apply_to_batch([raw, bboxes], 'crop')  # crop the raw images
        print "Saving crop regions..."
        if save_dir:
            for crop_boxes, filename in zip(cropped, filenames):
                for i, region in enumerate(crop_boxes):
                    imsave(join(save_dir, filename + '_{}.tiff'.format(i)), region)

        return cropped

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
