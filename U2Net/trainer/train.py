import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import signal

from trainer.config import *
from trainer.dataloader import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, ReLU, MaxPool2D, UpSampling2D
from model.u2net import *

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
    data_loading_mode = args.data_loading_mode

# Previewing, not necessary
img = cv2.imread('examples/skateboard.jpg')
img = cv2.resize(img, dsize=default_in_shape[:2], interpolation=cv2.INTER_CUBIC)
inp = np.expand_dims(img, axis=0)

# Overwrite the default optimizer
adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-08)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_file, save_weights_only=True, verbose=1)

def train():
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
        print('Saving state of model to %s' % weights_file)
        model.save_weights(str(weights_file))
    
    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('Training aborted early... Saving weights.')
        save_weights()
        exit(0)

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)
    # Configuration for GPU if possible 
    gpu_available = tf.test.is_gpu_available()
    if gpu_available:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            # start training
            print('Starting training with GPU')
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
                    print('[%05d] Loss: %.4f' % (e, loss))

                if save_interval and e > 0 and e % save_interval == 0:
                    save_weights()
    else:
        print('Starting training with CPU')
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
                print('[%05d] Loss: %.4f' % (e, loss))

            if save_interval and e > 0 and e % save_interval == 0:
                save_weights()

if __name__=="__main__":
    train()