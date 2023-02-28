import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal
import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from trainer.config import *
from trainer.dataloader import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from trainer.model.u2net import *


# Arguments
parser = argparse.ArgumentParser(description='U^2-NET Training')
parser.add_argument('--batch_size', default=None, type=int,
                    help='Training batch size')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Optimizer learning rate. Default: %s' % learning_rate)
parser.add_argument('--save_interval', default=None, type=int,
                    help='How many iterations between saving of model weights')
parser.add_argument('--weights_file', default=None, type=str,
                    help='Output location for model weights. Default: %s' % weights_file)
parser.add_argument('--data_loading_mode', default=1, type=int,
                    help='Loading dataset mode. zipped = 1 | bucket = 2 | fuse = 3 Default: %s' % 1)
parser.add_argument('--resume', default=None, type=str,
                    help="Resume training network from saved weights file. Leave as None to start new training.")
parser.add_argument('--bucket_name', default=None, type=str,
                    help="Bucket name for loading data and saving model into")
args = parser.parse_args()

if args.batch_size:
    batch_size = args.batch_size

if args.lr:
    learning_rate = args.lr

if args.save_interval:
    save_interval = args.save_interval

if args.weights_file:
    weights_file = weight_dir.joinpath(args.weights_file)

if not weight_dir.exists():
    weight_dir.mkdir()

if args.resume:
    resume = args.resume

if args.data_loading_mode:
    data_loading_mode = args.data_loading_mode or LOCAL_TEST

if args.bucket_name:
    BUCKET_NAME = args.bucket_name

# Previewing, not necessary
# img = cv2.imread('examples/skateboard.jpg')
# img = cv2.resize(img, dsize=default_in_shape[:2], interpolation=cv2.INTER_CUBIC)
# inp = np.expand_dims(img, axis=0)

# Overwrite the default optimizer
adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_file, save_weights_only=True, verbose=1)


print("==================== ARGS PARSER ========================")
print(args)
bucket_fuse_mode = BUCKET_NAME and data_loading_mode in [FROM_BUCKET, FROM_FUSE]
print(f"loading_mode: (passed, received) ({args.data_loading_mode}, {data_loading_mode})")
print(f"BUCKET NAME IS:{BUCKET_NAME} and Will I save to bucket ? {'Yes' if bucket_fuse_mode else 'No'}")
loss_metric = tf.keras.metrics.Mean()

def train():
    print('SCRIPT IS STARTED WITH ARGS')
    print(sys.argv)
    inputs = keras.Input(shape=default_in_shape)
    net = U2NET()
    out = net(inputs)
    model = keras.Model(inputs=inputs, outputs=out, name='u2netmodel')
    model.compile(optimizer=adam, loss=bce_loss, metrics=None)
    model.summary()

    if resume:
        print('Loading weights from %s' % resume)
        model.load_weights(resume)

    # setup the dataset if the user hasn't set it up yet
    image_dir, mask_dir = prepare_data_set(mode=data_loading_mode)

    # helper function to save state of model
    def save_weights():
        if resume:
            print('Resume Saving state of model to %s' % weights_file)
            model.save_weights(resume)
        else:
            print('Saving state of model to %s' % weights_file)
            model.save_weights(str(weights_file))
    
    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('Training aborted early... Saving weights.')
        save_weights()
        exit(0)

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)

    # Tensorboard for monitoring the training process 
    # Refer: https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=10,
        write_graph=True,
    )
    tensorboard.set_model(model)

    def named_logs(mode, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result
    tf.debugging.set_log_device_placement(True)

    for e in range(epochs):
        try:
            feed, out = load_training_batch(batch_size=batch_size, image_dir=image_dir, mask_dir=mask_dir)
            loss = model.train_on_batch(feed, out)
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            continue

        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e, loss), flush=True)

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()

    # # Configuration for GPU if possible 
    # gpu_available = len(tf.config.list_physical_devices('GPU')) > 0 
    # if gpu_available:
    #     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #     gpus = tf.config.list_logical_devices('GPU')
    #     strategy = tf.distribute.MirroredStrategy(gpus)
    #     with strategy.scope():
    #         # start training
    #         print('Starting training with GPU')
    #         for e in range(epochs):
    #             print('training on epoch', e)
    #             try:
    #                 feed, out = load_training_batch(batch_size=batch_size, image_dir=image_dir, mask_dir=mask_dir)
    #                 loss = model.train_on_batch(feed, out)
    #                 tensorboard.on_epoch_end(e, named_logs(model,loss))
    #             except KeyboardInterrupt:
    #                 save_weights()
    #                 return
    #             except ValueError:
    #                 continue

    #             if e % 10 == 0:
    #                 print('[%05d] GPU Loss: %.4f' % (e, loss), flush=True)

    #             if save_interval and e > 0 and e % save_interval == 0:
    #                 save_weights()
    # else:
    #     print('Starting training with CPU')
    #     for e in range(epochs):
    #         try:
    #             feed, out = load_training_batch(batch_size=batch_size, image_dir=image_dir, mask_dir=mask_dir)
    #             loss = model.train_on_batch(feed, out)
    #         except KeyboardInterrupt:
    #             save_weights()
    #             return
    #         except ValueError:
    #             continue

    #         if e % 10 == 0:
    #             print('[%05d] Loss: %.4f' % (e, loss), flush=True)

    #         if save_interval and e > 0 and e % save_interval == 0:
    #             save_weights()

if __name__=="__main__":
    train()