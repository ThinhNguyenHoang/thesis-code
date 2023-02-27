import pathlib

# Model
resume = None
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('u2net.h5')
default_in_shape = (320, 320, 3)
default_out_shape = (320, 320, 1)
data_loading_mode = 1
# Training
batch_size = 2
epochs = 10000
learning_rate = 0.001
save_interval = 1000


# Evaluation
output_dir = pathlib.Path('out')