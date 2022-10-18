# -*- coding: utf-8 -*-
"""
ResNet56-RoI - emotion detection
for small images
pretrainedmodel with fer2013 dataset

"""

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import os

class Predict:
    def __init__(self, model_weights_path):
        self.depth = 56
        self.input_shape = (48, 48, 1)
        resnet56 = Resnet56()
        self.model = resnet56.resnet_v2(input1=self.input_shape, input2=self.input_shape, depth=self.depth, num_classes=7)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=resnet56.lr_schedule(100)),
              metrics=['accuracy'])
        model_type = "ResNet_{}_RoI".format(self.depth)
        print("[*] Network: {}".format(model_type))
        self.model.load_weights(model_weights_path)



    def predict_emotion(self, ndarray_image_1, ndarray_image_2):
        num, width, height, channels = ndarray_image_1.shape
        assert width == 48, 'width is wrong. correct is 48. you entered {}'.format(width)
        assert height == 48, 'height is wrong. correct is 48. you entered {}'.format(height)
        assert channels == 1, 'channel is wrong. correct is 1. you entered {}'.format(channels)
        prediction = self.model.predict([ndarray_image_1, ndarray_image_2])
        return prediction


# hossein amini begin
class Resnet56:
    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 90:
            lr *= 0.5e-3
        elif epoch > 80:
            lr *= 1e-3
        elif epoch > 70:
            lr *= 1e-2
        elif epoch > 50:
            lr *= 1e-1
        return lr

    def resnet_layer(self, inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder
        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)
        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v2(self, input1, input2, depth, num_classes):
        """ResNet Version 2 Model builder [b]
        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        X_input1 = Input(input1)
        X_input2 = Input(input2)

        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x1 = self.resnet_layer(
          inputs=X_input1,
          num_filters=num_filters_in,
          kernel_size=3,
          strides=1,
          activation=None,
          batch_normalization=None,
          conv_first=True
        )

        x2 = self.resnet_layer(
          inputs=X_input2,
          num_filters=num_filters_in,
          kernel_size=3,
          strides=1,
          activation=None,
          batch_normalization=None,
          conv_first=True
        )

        x = Add()([x1, x2])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = self.resnet_layer(inputs=x,
                                num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
                y = self.resnet_layer(inputs=y,
                                num_filters=num_filters_in,
                                conv_first=False)
                y = self.resnet_layer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = tensorflow.keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=[X_input1, X_input2], outputs=outputs)
        return model
# hossein amini end
