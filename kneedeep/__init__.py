import os
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set_style('white')
from keras import backend as K

if os.name == 'nt':
    os.environ['KERAS_BACKEND'] = 'theano'
    K.set_image_data_format('channels_first')
else:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    K.set_image_data_format('channels_last')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import metrics