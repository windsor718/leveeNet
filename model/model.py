from keras.layers import Conv2D, Input, AveragePooling2D
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np


def build_block(input, ch):
    """
    A simple buidling block of two convolutional layers.

    Args:
        input (func): functional API input
        ch (int): filters
    
    Returns:
        func: updated functional API input
    """
    x = Conv2D(ch, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(ch, 3, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def build_block_NUS(input, ch, drop_rate=0.25):
    """
    A new building block architecture proposed by
    NUS research group.

    Args:
        input (func): functional API input
        ch (int): filters
    
    Returns:
        func: updated functional API input
    
    Notes:
        https://arxiv.org/pdf/1904.03392.pdf
    """
    x = Conv2D(ch, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(drop_rate)(x)
    x = Conv2D(ch, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return Dropout(drop_rate)(x)


def leveeNet(n_classes, image_size):
    """
    A simple CNN for image classification.

    Args:
        n_classes (int): number of classes
        image_size (tuple): [h, v, c]
    
    Returns:
        keras.models.Model
    """
    inp = Input(image_size)
    x = build_block(inp, 64)
    x = AveragePooling2D(2)(x)
    x = build_block(x, 128)
    x = AveragePooling2D(2)(x)
    x = build_block(x, 256)
    x = Dense(120, activation="relu")
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(input, x)
    return model