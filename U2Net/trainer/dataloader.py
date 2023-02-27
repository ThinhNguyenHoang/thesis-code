import os
import pathlib
import random
import numpy as np
import wget
import zipfile
import glob
import sys
import shutil
# sys.path.append('./')
from enum import Enum

from trainer.config import *
from PIL import Image 

cache = None
DATASET_NAME = 'DUTS-TR'
DATASET_IMAGE_FOLDER_NAME = 'DUTS-TR-Image'
DATASET_MASK_FOLDER_NAME = 'DUTS-TR-Mask'

# VARIABLES FOR TRAINING WITH BUCKETS
BUCKET_NAME = os.environ.get('BUCKET_NAME') or 'thesis-data-bucket'
def get_bucket_prefix(bucket_name=BUCKET_NAME):
    return f'/gcs/{bucket_name}'

BUCKET_PREFIX = f'/gcs/{BUCKET_NAME}'
DATASET_BUCKET_URI = f'{BUCKET_PREFIX}/datasets/'

FROM_ZIPPED = 1
FROM_BUCKET = 2
FROM_FUSE = 3
LOCAL_TEST = 4
# aborting wget leaves .tmp files everywhere >:(
def clean_dataloader():
    for tmp_file in glob.glob('*.tmp'):
        os.remove(tmp_file)

def try_remove_old_data(path_to_delete):
    try:
        if os.path.isdir(path_to_delete):
            shutil.rmtree(path_to_delete)
    finally:
        print('end of removing old dataset')

def download_duts_tr_dataset(dataset_dir=None, root_data_dir=None):
    # dataset_url = 'http://saliencydetection.net/duts/download/DUTS-TR.zip'
    # try_remove_old_data(root_data_dir.absolute())
    # pathlib.Path(root_data_dir.absolute()).mkdir(exist_ok=True)

    if dataset_dir.exists():
        print(f'DATSET ALREADY EXIST: {dataset_dir}')
        return

    print('Downloading DUTS-TR Dataset from %s...' % dataset_url)
    f = wget.download(dataset_url, out=str(root_data_dir.absolute()))
    if not pathlib.Path(f).exists():
        return

    print('Extracting dataset...')
    with zipfile.ZipFile(f, 'r') as zip_file:
        zip_file.extractall(root_data_dir.absolute())

    clean_dataloader()

def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))
# OUTPUT: (image_dir, output_dir)
def prepare_data_set(mode=FROM_ZIPPED):
    # Download and store in 'data'
    print(f'PREPARING DATASET WITH mode: {mode}')
    root_data_dir = pathlib.Path('data')
    dataset_dir = root_data_dir.joinpath(DATASET_NAME)
    if mode == LOCAL_TEST:
        print('TESTING WITH DATA PREPARED LOCALY')
        root = pathlib.Path('local_test_data')
        dataset_dir = root.joinpath(DATASET_NAME)
        image_dir = dataset_dir.joinpath(DATASET_IMAGE_FOLDER_NAME)
        mask_dir = dataset_dir.joinpath(DATASET_MASK_FOLDER_NAME)
        return (image_dir, mask_dir)
    elif mode == FROM_ZIPPED:
        print(f'DOWLOADING ZIPPED DS: {DATASET_BUCKET_URI}')
        download_duts_tr_dataset(dataset_dir,root_data_dir)
    elif mode == FROM_BUCKET:
        dataset_url = f'{DATASET_BUCKET_URI}'
        print(f'DOWNLOADING RAW DS FROM BUCKET: {DATASET_BUCKET_URI}')
        f = wget.download(dataset_url, out=str(root_data_dir.absolute()), bar=bar_custom)
        if not pathlib.Path(f).exists():
            return
    elif mode == FROM_FUSE:
        print(f'RUNNING WITH FUSE: {DATASET_BUCKET_URI}')
        print("WE ARE CURRENTLY AT")
        print(os.listdir())
        os.chdir(DATASET_BUCKET_URI)

        print("NOW WE ARE AT:")
        print(os.listdir())
        dataset_dir = pathlib.Path(DATASET_NAME)
        
    image_dir = dataset_dir.joinpath(DATASET_IMAGE_FOLDER_NAME)
    mask_dir = dataset_dir.joinpath(DATASET_MASK_FOLDER_NAME)
    print(f'RETURNIN (image_dir, mask_dir)=({image_dir},{mask_dir})')
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
    # x (320, 320)
    # expand_dims(x) => (320, 320, 1)
    # (320, 320, 3) and (320, 320, 1)
    return (np.asarray(img, dtype=np.float32), np.expand_dims(np.asarray(mask, dtype=np.float32), -1))

def load_training_batch(batch_size=12, in_shape=default_in_shape, out_shape=default_out_shape, image_dir=None, mask_dir=None):
    global cache
    if cache is None:
        cache = os.listdir(image_dir)
    if image_dir is None or mask_dir is None:
        raise ValueError('IMAGE and MASK dir cannot be None')
    imgs = random.choices(cache, k=batch_size)
    # [(img, mask)]
    image_list = [get_image_mask_pair(img_name=img, image_dir=image_dir, mask_dir=mask_dir, in_resize=default_in_shape, out_resize=default_out_shape) for img in imgs]
    # Loop and extract images in batch 
    tensor_in  = np.stack([i[0]/255. for i in image_list])
    # Loop and extract masks from batch
    tensor_out = np.stack([i[1]/255. for i in image_list])
    
    return (tensor_in, tensor_out)  
