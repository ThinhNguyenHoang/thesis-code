import tensorflow as tf
from config import Config
from tensorflow.keras import layers
from sub_models.MoINN.modules.coupling_layers import GLOWCouplingBlock
from sub_models.MoINN.modules.all_in_one_block import AllInOneBlock
import sub_models.u2net.eval as u2net_eval

from utils import debug

from typing import *
import sys
sys.path.append('./sub_models/tfdetection')

from sub_models.tfdetection.tfdet.model.backbone.resnet import RESNET_NBLOCKS_CONFIG, RESNET_OUTPUT_NAMES_CONFIG, ResNetArchOutputsDict
from sub_models.tfdetection.tfdet.model.backbone.resnet import wide_resnet50_2

def load_u2_net_eval(config):
    path = 'sub_models/u2net/weights/u2net.h5'
    # print("Loading saliency detector at:", path)
    # place_holder = tf.keras.Input(shape=(224,224,3))
    # u2_net = U2NET()
    # model = u2_net(place_holder)
    # model = tf.keras.Model(inputs=place_holder, outputs=u2_net, name='u2net_detector')
    model = u2net_eval.load_model_for_eval(path,config)
    return model

def load_saliency_detector(config:Config):
    saliency_detector = None
    if config.saliency_detector == 'u2net':
        saliency_detector = load_u2_net_eval(config)
    if not saliency_detector:
        raise NotImplementedError('Saliency detector is not supported')
    return saliency_detector

# Delay the application of the network
def load_encoder_arch(config, num_pool_layers):
    # Load the encoder in eval mode (pretrained)
    pool_dimensions = list()

    #
    input_size = config.input_size
    IMAGE_SIZE = (input_size, input_size, 3)
    inputs = layers.Input(shape=(IMAGE_SIZE))

    enc_arch = config.encoder_arch
    if 'resnet' in enc_arch:
        if enc_arch in ['resnet18', 'resnet34', 'resnet50']:
            raise NotImplementedError()
        elif enc_arch == 'wide_resnet50_2':
            outputs = wide_resnet50_2(inputs)
        else:
            raise NotImplementedError()
        # Calculate the pooling dimension to setup the decoders accordingly
        encoder = tf.keras.Model(inputs, outputs)
        resnet_stacks_config = RESNET_NBLOCKS_CONFIG[enc_arch]
        resnet_output_names: ResNetArchOutputsDict = RESNET_OUTPUT_NAMES_CONFIG[enc_arch][1:]
        # for idx, num_block in enumerate(resnet_stacks_config):
        #     layer_output_shape =resnet_get_layer_output_shape(model=encoder,stage_idx=idx + 1, block_idx=num_block, resnet_arch=enc_arch)
        #     pool_dimensions.append(layer_output_shape)
        pool_dimensions = [encoder.get_layer(name=layer_name).output_shape for layer_name in resnet_output_names]
        debug.debug_print(f"ENCODER[0]LAYER: {pool_dimensions[0]} ")
    return encoder, num_pool_layers, pool_dimensions

# Simple network for predicting parameters of the flows
def subnet_fc(meta, dimension_in, dimension_out):
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

def flow_with_condition(config: Config, input_dimension):
    condition_dimension = config.cond_vec_len
    pool_dimension = input_dimension[-1]
    inp = tuple(input_dimension[:])
    cond = tuple([*inp[:-1],condition_dimension])
    for _ in range(config.coupling_blocks):
        x = AllInOneBlock(inp, [cond],None,subnet_fc,permute_soft=False)
    return x

# target: Load a single flow (an decoder)
def load_decoder_arch(config: Config, input_dimension):
    decoder_arch = config.decoder_arch
    if config.debug:
        debug.debug_print(f'load_decoder_arch::input_dimenstion={input_dimension}')
    if decoder_arch == 'freia-flow':
        decoder = flow_without_condition(config, input_dimension)
    elif decoder_arch == 'freia-cflow':
        decoder = flow_with_condition(config, input_dimension)
    else:
        raise NotImplementedError('Unsupported decoder architecture')
    return decoder