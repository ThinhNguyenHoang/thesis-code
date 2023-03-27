import tensorflow as tf
import tensorflow.keras as keras
from config import Config
from tensorflow.keras import layers
from sub_models.MoINN.modules.coupling_layers import GLOWCouplingBlock
from sub_models.MoINN.modules.all_in_one_block import AllInOneBlock
import sub_models.u2net.eval as u2net_eval
import cv2
from utils.positional_encoding import TFPositionalEncoding2D
from utils.loss import get_logp, t2np
import numpy as np
from PIL import Image
from utils import debug

from typing import *
import sys
sys.path.append('./sub_models/tfdetection')

from sub_models.tfdetection.tfdet.model.backbone.resnet import RESNET_NBLOCKS_CONFIG, RESNET_OUTPUT_NAMES_CONFIG, ResNetArchOutputsDict
from sub_models.tfdetection.tfdet.model.backbone.resnet import wide_resnet50_2

class MainModel(keras.Model):
    def __init__(self, config:Config, subnet_fc):
        super().__init__()
        self.input_size = config.input_size
        self.IMG_SQUARE_SIZE = [self.input_size, self.input_size]
        self.IMG_SIZE = (self.input_size, self.input_size, 3)
        self.pool_layers = config.pool_layers
        self.encoder, self.num_pool_layer, self.pool_dimensions = self.load_encoder(config)
        self.subnet_fc = subnet_fc
        self.decoders = self.init_decoders(config)
        self.saliency_detector = self.load_saliency_detector(config)
        self.config = config

    def compile(self, optimizer, loss_fn=None, **kwargs):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    # ----------------------- ENCODER ----------------------------------------
    def load_encoder(self, config: Config):
        pool_dimensions = list()
        #
        inputs = layers.Input(shape=(self.IMG_SIZE), batch_size= config.batch_size)

        enc_arch = config.encoder_arch
        if 'resnet' in enc_arch:
            if enc_arch in ['resnet18', 'resnet34', 'resnet50']:
                raise NotImplementedError()
            elif enc_arch == 'wide_resnet50_2':
                outputs = wide_resnet50_2(inputs)
            else:
                raise NotImplementedError()
            # Calculate the pooling dimension to setup the decoders accordingly
            encoder = tf.keras.Model(inputs, outputs, name='feature_extractor')
            resnet_stacks_config = RESNET_NBLOCKS_CONFIG[enc_arch]
            resnet_output_names: ResNetArchOutputsDict = RESNET_OUTPUT_NAMES_CONFIG[enc_arch][1:]
            # for idx, num_block in enumerate(resnet_stacks_config):
            #     layer_output_shape =resnet_get_layer_output_shape(model=encoder,stage_idx=idx + 1, block_idx=num_block, resnet_arch=enc_arch)
            #     pool_dimensions.append(layer_output_shape)
            pool_dimensions = [encoder.get_layer(name=layer_name).output_shape for layer_name in resnet_output_names]
            debug.debug_print(f"ENCODER[0]LAYER: {pool_dimensions[0]} ")
        return encoder, config.pool_layers, pool_dimensions

    # ----------------------- DECODER ----------------------------------------
    def init_decoders(self, config: Config):
        decoders = [self.load_decoder(config, pool_dimension) for pool_dimension in self.pool_dimensions]
        return decoders

    # input_dimension: BxHxWxC
    def load_decoder(self, config:Config, input_dimension):
        decoder_arch = config.decoder_arch
        if config.debug:
            debug.debug_print(f'load_decoder_arch::input_dimenstion={input_dimension}')
        decoder = self.cflow(config, input_dimension)
        return decoder
    # input_dimension: BxHxWxC
    def cflow(self, config: Config, input_dimension):
        condition_dimension = config.cond_vec_len
        # pool_dimension = input_dimension[-1]
        sample_dimension = tuple(input_dimension[1:])
        cond = tuple([*sample_dimension[:-1],condition_dimension])

        inp_place_holder = layers.Input(sample_dimension, batch_size=config.batch_size)
        condition_place_holder = layers.Input(cond, batch_size=config.batch_size)
        # out = AllInOneBlock((input_dimension[-1],), [(cond[-1],)],{'feature_map_dims': inp, 'condition_dims': cond},self.subnet_fc,permute_soft=False)(inp_place_holder, condition_place_holder)
        blocks = [AllInOneBlock((input_dimension[-1],), [(cond[-1],)],{'feature_map_dims': sample_dimension, 'condition_dims': cond},self.subnet_fc,permute_soft=False) for _ in range(config.coupling_blocks)]
        z, log_jac_det = blocks[0](inp_place_holder, condition_place_holder)
        for block in blocks[1:]:
            z,log_jac_det = block(z, condition_place_holder)

        # return blocks
        decoder = keras.Model(inputs=[inp_place_holder, condition_place_holder],outputs=[z,log_jac_det])
        return decoder

    def flow(self,config, input_dimension):
        # coder
        condition_dimension = config.cond_vec_len
        coder = tf.keras.Sequential()
        if config.verbose:
            print('CNF coder', condition_dimension)
        for _ in range(config.coupling_blocks):
            # No condition
            coder.add(GLOWCouplingBlock(input_dimension,None, None, self.subnet_fc))
        return coder

    # ----------------------- SALIENCY DETECTOR ----------------------------------------
    def load_saliency_detector(self, config:Config):
        saliency_detector = None
        if config.saliency_detector == 'u2net':
            saliency_detector = self.load_u2_net_eval(config)
        if not saliency_detector:
            raise NotImplementedError('Saliency detector is not supported')
        return saliency_detector

    def load_u2_net_eval(self, config):
        path = 'sub_models/u2net/weights/u2net.h5'
        model = u2net_eval.load_model_for_eval(path,config)
        return model

    def call(self, data):
        feature_maps = self.encoder(data, training=False)
        print("num_of_multiscale_features:", len(feature_maps))

        # extract saliency map
        saliency_map = self.saliency_detector(data, Image.BICUBIC, training=False)[0] # BxHxWx1
        # input_img = tf.keras.utils.array_to_img(input_img)
        # saliency_map = u2net_eval.get_saliency_map(saliency_detector, img_copy)

        for idx, feature_map in enumerate(feature_maps):
            batch_size, height, width, depth = feature_map.shape
            # Interpolate - Resize the salience map to correct size
            instance_aware = tf.image.resize(saliency_map, [height,width])
            positional_encoding = TFPositionalEncoding2D(channels=self.config.cond_vec_len)(feature_map) # HxWxD
            # Ensure positional_encoding and instance_aware has the same shape
            assert instance_aware.shape == positional_encoding.shape, 'encoding and feature map should have same shape'
            condition_vec = tf.math.multiply(instance_aware, positional_encoding) # BxHxD -- D = dimensions of encodings
            decoder = self.decoders[idx]
            # FIBER PROCESSING
            # feature_map: BxHxC ---- C: channels of pooling layers
            # condition_vec: BxHxD ------ D: dimension of positional encoding
            squared_size = height * width # HxW
            num_features_vecs = batch_size * squared_size # Bx(HW)

            features = tf.reshape(feature_map, [num_features_vecs, depth])
            conditions = tf.reshape(condition_vec, [num_features_vecs, self.config.cond_vec_len])
            #
            N = self.config.N
            UNIT = num_features_vecs // N + int(num_features_vecs % N > 0)
            perm = tf.random.shuffle(np.arange(num_features_vecs))
            train_loss = 0
            for unit in range(UNIT):
                if unit < (UNIT -1):
                    idx = np.arange(unit * N, (unit + 1) * N)
                else:
                    idx = np.arange(unit*N, num_features_vecs)
                #
                perm_idx = tf.gather(perm, idx)
                feature_patch = tf.gather(features, perm_idx) # NxC ----- C:channels
                condition_patch = tf.gather(conditions, perm_idx) #  NxP ---------- P: pos_enc dimensitons
                with tf.GradientTape() as tape:
                    # z = feature_patch
                    # for block, blocks in enumerate(decoder):
                    #     # z, log_jac_det = decoder[block](z, [condition_patch,])
                    #     z, log_jac_det = decoder[block](z, [condition_patch,])
                    z, log_jac_det = decoder(feature_patch, [condition_patch,])
                    decoder_log_prob = get_logp(depth, z, log_jac_det)
                    log_prob = decoder_log_prob / depth
                    # Normalizing to be in range (0,1)
                    loss= -tf.math.log_sigmoid(log_prob)
                    mean_loss = tf.math.reduce_mean(loss)
                    self.add_loss(mean_loss)
                gradients = tape.gradient(mean_loss, decoder.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
                train_loss += tf.reduce_sum(loss)
                train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        return mean_train_loss
    def train_step(self, data):
        feature_maps = self.encoder(data, training=False)
        print("num_of_multiscale_features:", len(feature_maps))

        # extract saliency map
        saliency_map = self.saliency_detector(data, Image.BICUBIC, training=False)[0] # BxHxWx1
        # input_img = tf.keras.utils.array_to_img(input_img)
        # saliency_map = u2net_eval.get_saliency_map(saliency_detector, img_copy)

        for idx, feature_map in enumerate(feature_maps):
            batch_size, height, width, depth = feature_map.shape
            # Interpolate - Resize the salience map to correct size
            instance_aware = tf.image.resize(saliency_map, [height,width])
            positional_encoding = TFPositionalEncoding2D(channels=self.config.cond_vec_len)(feature_map) # HxWxD
            # Ensure positional_encoding and instance_aware has the same shape
            assert instance_aware.shape == positional_encoding.shape, 'encoding and feature map should have same shape'
            condition_vec = tf.math.multiply(instance_aware, positional_encoding) # BxHxD -- D = dimensions of encodings
            decoder = self.decoders[idx]
            # FIBER PROCESSING
            # feature_map: BxHxC ---- C: channels of pooling layers
            # condition_vec: BxHxD ------ D: dimension of positional encoding
            squared_size = height * width # HxW
            num_features_vecs = batch_size * squared_size # Bx(HW)

            features = tf.reshape(feature_map, [num_features_vecs, depth])
            conditions = tf.reshape(condition_vec, [num_features_vecs, self.config.cond_vec_len])
            #
            N = self.config.N
            UNIT = num_features_vecs // N + int(num_features_vecs % N > 0)
            perm = tf.random.shuffle(np.arange(num_features_vecs))
            train_loss = 0
            for unit in range(UNIT):
                if unit < (UNIT -1):
                    idx = np.arange(unit * N, (unit + 1) * N)
                else:
                    idx = np.arange(unit*N, num_features_vecs)
                #
                perm_idx = tf.gather(perm, idx)
                feature_patch = tf.gather(features, perm_idx) # NxC ----- C:channels
                condition_patch = tf.gather(conditions, perm_idx) #  NxP ---------- P: pos_enc dimensitons
                with tf.GradientTape() as tape:
                    # z = feature_patch
                    # for block, blocks in enumerate(decoder):
                    #     # z, log_jac_det = decoder[block](z, [condition_patch,])
                    #     z, log_jac_det = decoder[block](z, [condition_patch,])
                    z, log_jac_det = decoder(feature_patch, [condition_patch,])
                    decoder_log_prob = get_logp(depth, z, log_jac_det)
                    log_prob = decoder_log_prob / depth
                    # Normalizing to be in range (0,1)
                    loss= self.loss_fn(log_prob)
                    mean_loss = tf.math.reduce_mean(loss)
                gradients = tape.gradient(mean_loss, decoder.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
                train_loss += tf.reduce_sum(loss)
                train_count += len(loss)
        #
        mean_train_loss = train_loss / train_count
        return mean_train_loss

    def test_step(self, data):
        if self.config.verbose:
            print("Computing loss and scores on test set:")
        images, labels, ground_truths = data
        #
        cond_vec_lend = self.config.cond_vec_len
        height = list()
        width = list()
        image_list = list()
        gt_label_list = list()
        gt_mask_list = list()
        test_dist = [list() for layer in self.pool_layers]
        test_loss = 0.0
        test_count = 0
        for i, (image, label, ground_truth_mask) in enumerate(data):
            if self.config.viz:
                image_list.extend()
            gt_label_list.extend(label)
            gt_mask_list.extend(ground_truth_mask)
            feature_maps = self.encoder(image)
            saliency_map = self.saliency_detector(image, Image.BICUBIC)[0] # BxHxWx1
            for layer, feature_map in enumerate(feature_maps):
                batch_size, height, width, depth = feature_map.shape
                # Interpolate - Resize the salience map to correct size
                instance_aware = tf.image.resize(saliency_map, [height,width])
                positional_encoding = TFPositionalEncoding2D(channels=self.config.cond_vec_len)(feature_map) # HxWxD
                # Ensure positional_encoding and instance_aware has the same shape
                assert instance_aware.shape == positional_encoding.shape, 'encoding and feature map should have same shape'
                condition_vec = tf.math.multiply(instance_aware, positional_encoding) # BxHxD -- D = dimensions of encodings
                decoder = self.decoders[idx]
                # FIBER PROCESSING
                # feature_map: BxHxC ---- C: channels of pooling layers
                # condition_vec: BxHxD ------ D: dimension of positional encoding
                squared_size = height * width # HxW
                num_features_vecs = batch_size * squared_size # Bx(HW)

                features = tf.reshape(feature_map, [num_features_vecs, depth])
                conditions = tf.reshape(condition_vec, [num_features_vecs, self.config.cond_vec_len])
                #
                N = self.config.N
                UNIT = num_features_vecs // N + int(num_features_vecs % N > 0)
                for unit in range(UNIT):
                    if unit < (UNIT -1):
                        idx = np.arange(unit * N, (unit + 1) * N)
                    else:
                        idx = np.arange(unit*N, num_features_vecs)
                    #
                    feature_patch = tf.gather(features, idx) # NxC ----- C:channels
                    condition_patch = tf.gather(conditions, idx) #  NxP ---------- P: pos_enc dimensitons
                    z = feature_patch
                    for _ in enumerate(decoder):
                        z, log_jac_det = decoder(z, [condition_patch,])
                    decoder_log_prob = get_logp(depth, z, log_jac_det)
                    log_prob = decoder_log_prob / depth
                    # Normalizing to be in range (0,1)
                    loss= -tf.math.log_sigmoid(log_prob)
                    test_loss += tf.reduce_sum(loss)
                    test_count += len(loss)
                    test_dist[layer] = test_dist[layer] + log_prob
            #
        mean_test_loss = test_loss / test_count
        return mean_test_loss
