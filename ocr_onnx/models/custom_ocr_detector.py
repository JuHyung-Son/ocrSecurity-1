import sys
sys.path.append('../')
#from models.ops import *

import numpy as np
import tensorflow as tf
from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D,MaxPooling2D, MaxPool2D, Activation, concatenate, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects, get_file
from tensorflow import keras
from models.util_rec import _transform

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

def make_vgg_block(x, filters, n, prefix, pooling=True):
    x = keras.layers.Conv2D(filters=filters,
                            strides=(1, 1),
                            kernel_size=(3, 3),
                            padding='same',
                            name=f'{prefix}.{n}')(x)
    x = keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, axis=-1,
                                        name=f'{prefix}.{n+1}')(x)
    x = keras.layers.Activation('relu', name=f'{prefix}.{n+2}')(x)
    if pooling:
        x = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                      padding='valid',
                                      strides=(2, 2),
                                      name=f'{prefix}.{n+3}')(x)
    return x

def build_vgg_backbone(inputs):
    x = make_vgg_block(inputs, filters=64, n=0, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=64, n=3, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=7, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=10, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=256, n=14, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=17, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=20, pooling=True, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=24, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=27, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=30, pooling=True, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=34, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=37, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=40, pooling=True, prefix='basenet.slice4')
    vgg = keras.models.Model(inputs=inputs, outputs=x)
    return [
        vgg.get_layer(slice_name).output for slice_name in [
            'basenet.slice1.12',
            'basenet.slice2.19',
            'basenet.slice3.29',
            'basenet.slice4.39',
        ]
    ]

def upconv(x, n, filters):
    x = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=f'upconv{n}.conv.0')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.1')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.2')(x)
    x = keras.layers.Conv2D(filters=filters // 2,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            name=f'upconv{n}.conv.3')(x)
    # x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.4')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.5')(x)
    return x

# class UpsampleLike(keras.layers.Layer):
#     """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
#     """
#
#     # pylint:disable=unused-argument
#     def call(self, inputs, **kwargs):
#         source, target = inputs
#         target_shape = keras.backend.shape(target)
#         if keras.backend.image_data_format() == 'channels_first':
#             raise NotImplementedError
#         else:
#             # pylint: disable=no-member
#             return tf.compat.v1.image.resize_bilinear(source,
#                                                       size=(target_shape[1], target_shape[2]),
#                                                       half_pixel_centers=True)
#
#     def compute_output_shape(self, input_shape):
#         if keras.backend.image_data_format() == 'channels_first':
#             raise NotImplementedError
#         else:
#             return (input_shape[0][0], ) + input_shape[1][1:3] + (input_shape[0][-1], )

def build_model(weights_path: str = None):
    inputs = keras.layers.Input((512, 512, 1),name='input')
    s1, s2, s3, s4 = build_vgg_backbone(inputs)
    s5 = keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same',
                                   name='basenet.slice5.0')(s4)

    s5 = keras.layers.Conv2D(1024,
                             kernel_size=(3, 3),
                             padding='same',
                             strides=1,
                             dilation_rate=6,
                             name='basenet.slice5.1')(s5)
    s5 = keras.layers.Conv2D(1024,
                             kernel_size=1,
                             strides=1,
                             padding='same',
                             name='basenet.slice5.2')(s5)
    s5 = keras.layers.Concatenate(name='cont1')([s5, s4])
    s5 = upconv(s5, n=1, filters=512)
    # target_shape = keras.backend.shape(s3)
    #
    # s5 = tf.image.resize(s5,size=(target_shape[1], target_shape[2]),name='resize1')

    # compat.v1.image.resize_bilinear(s5,
    #                                    size=(target_shape[1], target_shape[2]),
    #                                    half_pixel_centers=True,name='resize1')

    # s5 = keras.layers.Concatenate(name='cont2')([s5, s3])


    s5 = keras.layers.Conv2D(10,
                             kernel_size=1,
                             strides=1,
                             padding='same',
                             name='basenet.slice5.3')(s5)
    s5 = Activation('relu', name='relu_conv10')(s5)
    s5 = GlobalAveragePooling2D()(s5)
    y = Activation('softmax', name='loss')(s5)
    model = keras.models.Model(inputs=inputs, outputs=y)

    return model
