import os, time
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
from model import load_encoder_arch, load_decoder_arch,load_saliency_detector, subnet_fc
from utils.positional_encoding import TFPositionalEncoding2D
from utils.loss import get_logp, t2np
from config import Config
import sub_models.u2net.eval as u2net_eval
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution 
from utils.loss import negative_log_likelihood
disable_eager_execution()

from main_model import MainModel
# condition = saliency_map * positional_encoding
def get_condition_vec(positional_embedding, saliency_map):
    return tf.math.multiply(positional_embedding, saliency_map)

def train_meta_epoch(config, epoch, loader, encoder, saliency_detector, decoders, optimizer, pool_layers, N):
    positional_encoding_dim = config.condition_vec # CONDITION_VEC_LEN
    num_pool_layers = config.pool_layers
    num_iteration = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(config.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i in range(num_iteration):
            # extract feature maps
            image = next(iterator) # !DOUBLE_CHECK: iterator here working ??

            # saliency_map
            saliency_map = saliency_detector.predict(image)

            # features map
            feature_maps = encoder.predict()
            for idx, layer in enumerate(pool_layers):
                feature_map = tf.stop_gradient(feature_maps[layer])
                #
                batch_size, height, width, channel = feature_map.size() # BxHxWxD  D == C in original impl
                image_size = height * width # S=HxW
                embed_size = batch_size * image_size # E = BxHW
                # !DOUBLE CHECK THE ENCODING AND BATCHSIZE IS CORRECT
                # postional encoding
                cond_vec_len = config.cond_vec_len
                position_embedder = TFPositionalEncoding2D(cond_vec_len)
                positional_embedding = position_embedder(tf.zeros((1, height, width, channel))) # 1xHxWxD
                positional_embedding = tf.tile(positional_embedding, [batch_size, 1, 1, 1]) # BxHxWxD

                #
                feature_map = tf.reshape(tf.transpose(tf.reshape(batch_size, channel, image_size), [0, 2, 1]), embed_size, channel)
                positional_embedding = tf.reshape(tf.transpose(tf.reshape(batch_size, cond_vec_len, image_size), [0, 2, 1]), embed_size, cond_vec_len)

                rng = np.random.default_rng()
                perm = rng.shuffle(np.arange(embed_size))
                decoder = decoders[layer] # Keras Flow model
                #
                FIB = embed_size // N
                assert FIB > 0, 'Not enough fiber, please decrease batchsize or N'
                for f in range(FIB):
                    idx = np.arange(f*N, (f+1)*N)
                    condition_patch = positional_embedding[perm[idx]] # NxP
                    feature_map = feature_map[perm[idx]] # NxC
                    if 'cflow' in config.decoder_arch: # ! CHECK CONFIG VALUE
                        z, log_jac_det = decoder(feature_map, [condition_patch])
                    else:
                        z, log_jac_det = decoder(feature_map)
                    #
                    decoder_log_prob = get_logp(channel, z, log_jac_det)
                    log_prob = decoder_log_prob / channel
                    # Normalizing to be in range (0,1)
                    loss= -tf.math.log_sigmoid(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
    #
    mean_train_loss = train_loss / train_count
    if config.verbose:
        print(f'Epoch: {epoch}.{sub_epoch} train_loss: {mean_train_loss}, lr={1}')



def build_general_arch(config):
    #
    img_size = config.input_size
    IMG_SIZE = [img_size, img_size]
    #
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
    #
    pool_layers = config.pool_layers
    print('building keras model', pool_layers)
    # Load encoder in evaluate mode and model building
    encoder, pool_layers, pool_dimensions = load_encoder_arch(config, pool_layers)

    decoders =  [load_decoder_arch(config, pool_dimension) for pool_dimension in pool_dimensions]

    saliency_detector = load_saliency_detector(config)

    input_img = keras.layers.Input(shape=(*IMG_SIZE,3))
    # extract feature maps
    feature_maps = encoder(input_img)
    print("num_of_multiscale_features:", len(feature_maps))

    # extract saliency map
    saliency_map = saliency_detector(input_img, Image.BICUBIC)[0] # BxHxWx1
    # input_img = tf.keras.utils.array_to_img(input_img)
    # saliency_map = u2net_eval.get_saliency_map(saliency_detector, img_copy)

    for idx, feature_map in enumerate(feature_maps):
        batch_size, height, width, depth = feature_map.shape
        # Interpolate - Resize the salience map to correct size
        # instance_aware = tf.reshape(saliency_map, [batch_size,height, width,-1])
        instance_aware = tf.image.resize(saliency_map, [height,width])
        positional_encoding = TFPositionalEncoding2D(channels=config.cond_vec_len)(feature_map) # HxWxD
        # Ensure positional_encoding and instance_aware has the same shape
        # assert instance_aware.shape == positional_encoding.shape, 'encoding and feature map should have same shape'
        condition_vec = tf.math.multiply(instance_aware, positional_encoding) # BxHxD -- D = dimensions of encodings
        decoder = decoders[idx]
        # FIBER PROCESSING 
        # feature_map: BxHxC ---- C: channels of pooling layers
        # condition_vec: BxHxD ------ D: dimension of positional encoding
        squared_size = height * width # HxW
        num_features_vecs = batch_size * squared_size # Bx(HW)

        features = tf.reshape(feature_map, [num_features_vecs, depth])
        conditions = tf.reshape(condition_vec, [num_features_vecs, config.cond_vec_len])
        #
        N = 240
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
                if 'cflow' in config.decoder_arch: #!CHECK CONFIG VALUE
                    # inp = layers.Concatenate([feature_patch, condition_patch])
                    z, log_jac_det = decoder([feature_patch, [condition_patch,]])
                else:
                    z, log_jac_det = decoder(feature_patch)
                decoder_log_prob = get_logp(depth, z, log_jac_det)
                log_prob = decoder_log_prob / depth
                # Normalizing to be in range (0,1)
                loss= -tf.math.log_sigmoid(log_prob)
                loss = tf.math.reduce_mean(loss)
                gradients = tape.gradient(loss, decoder.trainable_weights)
            optimizer.apply_gradients(zip(gradients, decoder.trainable_weights))
            # train_loss += t2np(tf.reduce_sum(loss))
            # train_count += len(loss)

    model = keras.Model(inputs=input_img, outputs=[decoder.output for decoder in decoders])
    model.summary()
    return model

def train_with_keras(config):
    # # Get the saliency image: (black and white)
    # model = build_general_arch(config)
    # # optimizer
    # optimizer = keras.optimizers.Adam(learning_rate=0.01)
    
    # # Dataset preparation
    # if config.dataset == 'plant_village':
    #     train_dataset = tfds.load('plant_village', split='train', shuffle_files=True)
    #     # test_dataset = tfds.load('plant_village', split='test')
    # else:
    #     raise NotImplementedError("NOT SUPPORTED DATASET")

    # # Network hyperparameters
    # N = 256
    # print(f'train datasete info: len={len(train_dataset)}')
    # print(f'test datasete info: len={len(test_dataset)}')
    pass

def train_with_keras_subclass(config: Config, dataset=None, loader=None):
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    img_size = config.input_size
    # inp = keras.layers.Input(shape=(img_size,img_size,3), batch_size=config.batch_size)
    model = MainModel(config=config, subnet_fc=subnet_fc)
    # model = keras.Model(inputs=inp, outputs=[decoder.output for decoder in main.decoders.layers])
    model.compile(optimizer=adam, loss_fn=negative_log_likelihood)
    # model.build(input_shape=(img_size, img_size, 3), batch)
    model.fit(loader, epochs=config.sub_epochs, callbacks=[])
    
def train(config):
    # Get the saliency image: (black and white)
    pool_layers = config.pool_layers
    print('Number of pooling ', pool_layers)
    # Load encoder in evaluate mode and model building
    # Dataset preparation
    if config.dataset == 'plant_village':
        train_dataset = tfds.load('plant_village', split='train', shuffle_files=True)
        # test_dataset = tfds.load('plant_village', split='test')
    else:
        raise NotImplementedError("NOT SUPPORTED DATASET")

    # Network hyperparameters
    N = 256
    print(f'train datasete info: len={train_dataset.cardinality()}')
    # print(f'test datasete info: len={len(test_dataset)}')

    # train_with_keras_subclass(config, loader=train_dataset)
    
    model = build_general_arch(config)