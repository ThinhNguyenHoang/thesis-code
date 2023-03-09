import tensorflow as tf
from tensorflow.keras import layers
from sub_models.MoINN.modules.all_in_one_block import *
from sub_models.MoINN.modules.coupling_layers import GLOWCouplingBlock

from typing import *

from sub_models.tfdetection.tfdet.model.backbone.resnet import RESNET_NBLOCKS_CONFIG
from sub_models.tfdetection.tfdet.model.backbone.resnet import wide_resnet50_2

def load_saliency_detetor(config):
    return config.saliency_detector
def compute_conv_layer_name(stage_idx, block_idx, block_type):
    conv_layer_name = ''
    # Watch resnet.py for more context
    if block_type == 'bottleneck':
        conv_layer_name = 'conv3'
    elif block_type == 'basic':
        conv_layer_name = 'conv2'
    else:
        raise NotImplementedError()
    return f'stage{stage_idx}_block{block_idx}_{conv_layer_name}'


# Name of each layers = stage{idx}_block{idx}_name
# Ex: stage1_block1_conv3
def resnet_get_layer_output_shape(model, stage_idx:int, block_idx:int, resnet_arch: str):
    block_type = 'bottleneck' if 'wide' in resnet_arch else 'basic'
    # 
    return model.get_layer(name=compute_conv_layer_name(stage_idx, block_idx, block_type))


# Delay the application of the network
def load_encoder_arch(config, num_pool_layers):
    # Load the encoder in eval mode (pretrained)
    pool_dimensions = list()

    inputs = layers.Input()

    enc_arch = config.encoder_architecture
    if 'resnet' in enc_arch:
        if enc_arch in ['resnet18', 'resnet34', 'resnet50']:
            raise NotImplementedError()
        elif enc_arch != 'wide_resnet50_2':
            outputs = wide_resnet50_2(inputs)
        else:
            raise NotImplementedError()
        # Calculate the pooling dimension to setup the decoders accordingly
        encoder = tf.keras.Model(inputs, outputs)
        resnet_stacks_config = RESNET_NBLOCKS_CONFIG[enc_arch]
        for idx, num_block in enumerate(resnet_stacks_config):
            layer_output_shape =resnet_get_layer_output_shape(model=encoder,stage_idx=idx + 1, block_idx=num_block, resnet_arch=enc_arch)
            pool_dimensions.append(layer_output_shape)
    return encoder, num_pool_layers, pool_dimensions

# Simple network for predicting parameters of the flows
def subnet_fc(dimension_in, dimension_out):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(dimension_in,)))
    model.add(layers.Dense(2*dimension_in,activation=None))
    model.add(layers.ReLU())
    model.add(layers.Dense(dimension_out))
    return model

def flow_without_condition(config,input_dimension):
    # coder
    condition_dimension = config.cond_vec_len
    coder = tf.keras.Sequential()
    if config.verbose:
        print('CNF coder', condition_dimension)
    for _ in range(config.coupling_blocks):
        # No condition
        coder.add(GLOWCouplingBlock(input_dimension,None, None, subnet_fc))
    
    return coder

def flow_with_condition(config, input_dimension):
    condition_dimension = config.cond_vec_len
    coder = tf.keras.Sequential()
    if config.verbose:
        print('CNF coder', condition_dimension)
    for _ in range(config.coupling_blocks):
        coder.add(GLOWCouplingBlock(input_dimension,condition_dimension, None, subnet_fc))
    return coder

# target: Load a single flow (an decoder)
def load_decoder_arch(config, input_dimension):
    decoder_arch = config.decoder_arch
    if decoder_arch == 'freia-flow':
        decoder = flow_without_condition(config, input_dimension)
    elif decoder_arch == 'freia-clow':
        decoder = flow_with_condition(config, input_dimension)
    else:
        raise NotImplementedError('Unsupported decoder architecture')
    return decoder