""" Tensorflow Keras SWA Object
"""

#import tensorflow.keras.backend as K
from torchbearer import Trial

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from .callback import create_swa_callback_class

SWA = create_swa_callback_class(Trial, Callback, BatchNormalization)
