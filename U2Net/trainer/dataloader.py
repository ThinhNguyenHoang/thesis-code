import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob
import sys
# sys.path.append('./')
from enum import Enum

from trainer.config import *
from PIL import Image 

cache = None
DATASET_NAME = 'DUTS-TR'
DATASET_IMAGE_FOLDER_NAME = 'DUTS-TR-Image'
DATASET_MASK_FOLDER_NAME = 'DUTS-TR-Mask'

# VARIABLES FOR TRAINING WITH BUCKETS
BUCKET_NAME = os.environ.get('BUCKET_NAME')
DATASET_BUCKET_URI = f'/gcs/{BUCKET_NAME}/datasets/'
class DataLoadingMode(Enum):
    FROM_ZIPPED = 1
    FROM_BUCKET = 2
    FROM_FUSE = 3
# aborting wget leaves .tmp files everywhere >:(
def clean_dataloader():
    for tmp_file in glob.glob('*.tmp'):
        os.remove(tmp_file)

def download_duts_tr_dataset(dataset_dir=None, root_data_dir=None):
    dataset_url = 'http://saliencydetection.net/duts/download/DUTS-TR.zip'
    if dataset_dir.exists():
        return

    print('Downloading DUTS-TR Dataset from %s...' % dataset_url)
    f = wget.download(dataset_url, out=str(root_data_dir.absolute()))
    if not pathlib.Path(f).exists():
        return

    print('Extracting dataset...')
    with zipfile.ZipFile(f, 'r') as zip_file:
        zip_file.extractall(root_data_dir.absolute())

    clean_dataloader()
# OUTPUT: (image_dir, output_dir)
def prepare_data_set(mode=DataLoadingMode.FROM_ZIPPED):
    # Download and store in 'data'
    root_data_dir = pathlib.Path('data')
    dataset_dir = root_data_dir.joinpath(DATASET_NAME)
    if mode is DataLoadingMode.FROM_ZIPPED:
        download_duts_tr_dataset(dataset_dir,root_data_dir)
    elif mode is DataLoadingMode.FROM_BUCKET:
        dataset_url = f'{DATASET_BUCKET_URI}'
        f = wget.download(dataset_url, out=str(root_data_dir.absolute()))
        if not pathlib.Path(f).exists():
            return
    elif mode is DataLoadingMode.FROM_FUSE:
        dataset_dir = pathlib.Path(DATASET_NAME)

    image_dir = dataset_dir.joinpath(DATASET_IMAGE_FOLDER_NAME)
    mask_dir = dataset_dir.joinpath(DATASET_MASK_FOLDER_NAME)
    return (image_dir, mask_dir)

def format_input(input_image):
    assert(input_image.size == default_in_shape[:2] or input_image.shape == default_in_shape)
    inp = np.array(input_image)
    if inp.shape[-1] == 4:
        input_image = input_image.convert('RGB')
    return np.expand_dims(np.array(input_image)/255., 0)

def get_image_mask_pair(img_name,  image_dir, mask_dir, in_resize=None, out_resize=None, augment=True,):
    in_img = image_dir.joinpath(img_name)
    out_img = mask_dir.joinpath(img_name.replace('jpg', 'png'))

    if not in_img.exists() or not out_img.exists():
        return None

    img  = Image.open(image_dir.joinpath(img_name))
    mask = Image.open(out_img)

    if in_resize:
        img = img.resize(in_resize[:2], Image.BICUBIC)
    
    if out_resize:
        mask = mask.resize(out_resize[:2], Image.BICUBIC)

    # the paper specifies the only augmentation done is horizontal flipping.
    if augment and bool(random.getrandbits(1)):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def load_training_batch(batch_size=12, in_shape=default_in_shape, out_shape=default_out_shape, image_dir=None, mask_dir=None):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    
    imgs = random.choices(cache, k=batch_size)
    image_list = [get_image_mask_pair(img, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    
    tensor_in  = np.stack([i[0]/255. for i in image_list])
    tensor_out = np.stack([i[1]/255. for i in image_list])
    
    return (tensor_in, tensor_out)  

print(f"TEST CONFIG {batch_size}")