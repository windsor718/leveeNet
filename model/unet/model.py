from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam


def u_Net(n_classes, input_size):
    """
    u_net model definition.

    Args:
        input_size (tuple): [height, width, n_channels]

    Notes:
        comment terminology:

        conv1 -----------------stage 1----------------- conv1_up
            conv2 -------------stage 2-------------- conv2_up
                conv3 ---------stage 3--------- conv3_up
                    conv4 -----stage 4------ conv4_up
                        conv5 -stage 5- conv5_up
    """
    inputs = Input(input_size)

    # stage 1
    conv1_enc = conv2d_uNet(inputs, 64)

    # stage 2
    pool2_enc = MaxPooling2D(2)(conv1_enc)
    conv2_enc = conv2d_uNet(pool2_enc, 128)

    # stage 3
    pool3_enc = MaxPooling2D(2)(conv2_enc)
    conv3_enc = conv2d_uNet(pool3_enc, 256)

    # stage 4
    pool4_enc = MaxPooling2D(2)(conv3_enc)
    conv4_enc = conv2d_uNet(pool4_enc, 512)

    # stage 5
    pool5 = MaxPooling2D(2)(conv4_enc)
    conv5 = conv2d_uNet(pool5, 1024)

    # stage 4
    upconv4 = concatenate([conv4_enc,
                           Conv2DTranspose(512, 2,
                                           strides=2,
                                           padding="same")(conv5)])
    conv4_dec = conv2d_uNet(upconv4, 512)

    # stage 3
    upconv3 = concatenate([conv3_enc,
                           Conv2DTranspose(256, 2,
                                           strides=2,
                                           padding="same")(conv4_dec)])
    conv3_dec = conv2d_uNet(upconv3, 256)

    # stage 2
    upconv2 = concatenate([conv2_enc,
                           Conv2DTranspose(128, 2,
                                           strides=2,
                                           padding="same")(conv3_dec)])
    conv2_dec = conv2d_uNet(upconv2, 128)

    # stage 1
    upconv1 = concatenate([conv1_enc,
                           Conv2DTranspose(64, 2,
                                           strides=2,
                                           padding="same")(conv2_dec)])
    conv1_dec = conv2d_uNet(upconv1, 64)

    # output
    outputs = Conv2D(n_classes, 1, activation="sigmoid")(conv1_dec)

    return Model(inputs, outputs)


def conv2d_uNet(input, nfilters,
                kernel_size=3):
    """
    convolutional block in the U-Net arch.

    Args:
        input (keras tensor): input tensor
        nfilters (int): number of filters
    """
    x = Conv2D(nfilters, kernel_size,
               activation="relu", padding="same")(input)
    x = BatchNormalization()(x)
    # x = Dropout(dropout_rate)(x)
    x = Conv2D(nfilters, kernel_size,
               activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    return x
