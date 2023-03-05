# Test implementation of resent in keras


# import tensorflow as tf
# from typing import *
# from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D

# # Buckets URL to retrieve pretrained models
# model_urls = {

# }

# PADDING_MODE = 'SAME'


# def batch_norm(axis=-1, momentum=0.9, epsilon=1e-5, **kwargs):
#     return BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon, **kwargs)


# def conv3x3(x, filters: int, strides: int, padding='valid', groups=1,
#             dilation_rate: Union[Tuple[int, int], int] = (1, 1)):
#     return Conv2D(x, filters=filters, kernel_size=3, strides=strides, use_bias=False, dilation_rate=dilation_rate, padding=padding, groups=groups)


# def conv1x1(x, filters: int, strides: int, padding='valid', groups=1,
#             dilation_rate: Union[Tuple[int, int], int] = (1, 1)):
#     return Conv2D(x, filters=filters, kernel_size=1, strides=strides, use_bias=False, dilation_rate=dilation_rate, padding=padding, groups=groups)


# def basic_block(x,filters, strides, downsample, groups, base_width, dilation, norm_layer):
#     out = conv3x3(x)