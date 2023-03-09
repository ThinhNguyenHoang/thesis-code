import os, time
import tensorflow as tf
import tensorflow_datasets as tfds
import tf.keras as keras
from model import load_encoder_arch, load_decoder_arch
from utils.positional_encoding import TFPositionalEncoding2D
from utils.loss import get_logp, t2np
import numpy as np

# Arguments
# def encoding_module_eval(features, model):
#     anomaly_feature = model.encoder.encode(features)
#     # Calculate loglikelyhood and estimate density
#     score = get_score(anomaly_feature)
#     pass

def eval_one_batch(input, model):
    # Calculate the salient object:
    salient_obj_img  = model.saliency_detector.eval(input)
    features = model.feature_extrator.eval(input) # L feature map
    
    pass
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
                feature_map = tf.reshape(tf.tranpose(tf.reshape(batch_size, channel, image_size), [0, 2, 1]), embed_size, channel)
                positional_embedding = tf.reshape(tf.tranpose(tf.reshape(batch_size, cond_vec_len, image_size), [0, 2, 1]), embed_size, cond_vec_len)

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



def train(config):
    # Get the saliency image: (black and white)
    pool_layers = config.pool_layers
    print('Number of pooling ', pool_layers)
    # Load encoder in evaluate mode and model building
    encoder, pool_layers, pool_dimensions = load_encoder_arch(config, pool_layers)

    decoders =  [load_decoder_arch(config, pool_dimension) for pool_dimension in pool_dimensions]

    # optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    
    # Dataset preparation
    if config.dataset == 'plant_village':
        train_dataset = tfds.load('plant_village', split='train', shuffle_files=True)
        test_dataset = tfds.load('plant_village', split='test')
    else:
        raise NotImplementedError("NOT SUPPORTED DATASET")

    # Network hyperparameters
    N = 256
    print(f'train datasete info: len={len(train_dataset)}')
    print(f'test datasete info: len={len(test_dataset)}')
    # stats: AUROC RAUPRO ...
    
    #
    if config.action_type == 'norm-test':
        config.meta_epochs = 1
    for epoch in range(config.meta_epochs):
        if config.action_type == 'testing' and config.checkpoint:
            load_weights(encoder,decoders, config.checkpoint)
        elif config.action_type == 'training':
            print("TRAIN META EPOCH: ", epoch)
            train_meta_epoch(config, epoch, train_dataset, encoder, )
        else:
            raise NotImplementedError("Unsupported mode")
