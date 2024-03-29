# from typing import TypedDict, Union
import argparse

INPUT_SIZE = 224 # 256x256
BATCH_SIZE = 8

COUPLING_BLOCKS = 1
POOLING_LAYERS = 4
class Config(argparse.Namespace):
    action_type: str
    verbose: bool

    # Dataset
    dataset: str
    class_name: str
    bucket_save_path: str
    input_size: int
    # save AUPRO metric
    pro: bool
    
    # General Config
    meta_epochs: int
    sub_epochs: int
    batch_size: int
    N: int
    
    learning_rate: float
    dropout: float
    # model arch config
    # Saliency config
    saliency_detector: str
    feature_extractor: str # String like: wide_resnet_50_2 | densenet,...
    encoder_arch: str
    decoder_arch: str # freia-cflow | freia-flow
    local_saving_path: str
    output_bucket_path: str # Should be: /gcs/...
    pool_layers: int
    coupling_blocks: int
    #
    cond_vec_len: int # dimension of positional encoding
    # Debug printing config
    verbose: bool # Should only print debugs if true
    viz: bool # save test image (like saliency maps)
    debug: bool

    run_name: str 
def get_args():
    parser = argparse.ArgumentParser(description='Saliency_Flow')
    # General Mode
    parser.add_argument("--run-name", default='model', type=str, metavar='T',
                        help='anything to describe the run')
    parser.add_argument("--action-type", default='training', type=str, metavar='T',
                        help='training/testing (default: training)')
    parser.add_argument('--verbose', default=True, type=bool, metavar='G',
                        help='printing additional info message (default: true)')
    parser.add_argument('--debug', default=True, type=bool, metavar='G',
                        help='printing debug message (default: true)')
    # Model meta data config
    parser.add_argument('--dataset', default='plant_village', type=str, metavar='D', help='dataset name: plant_village (default: plant_village)')
    parser.add_argument('--class_name', default='tomato', type=str, metavar='D', help='class_name of a class inside the dataset')
    parser.add_argument('--bucket_save_path', default='FIX_DEFAULT', type=str, metavar='D',
                        help='where to save model output to')
    parser.add_argument('--input-size', default=INPUT_SIZE, type=int, metavar='C',
                        help='image resize dimensions (default: 224)')

    # Evaluation config
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    # Training Args
    parser.add_argument('--meta-epochs', type=int, default=25, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub-epochs', type=int, default=8, metavar='N',
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('-bs', '--batch-size', default=BATCH_SIZE, type=int, metavar='B',
                        help='train batch size (default: 12)')
    
    parser.add_argument('-N', '--N', default=240, type=int, metavar='N',
                        help='fibere batches: default 240')

    parser.add_argument('-lr','--learning-rate', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    # Model Architecture
    parser.add_argument('--saliency_detector', default='u2net', type=str, metavar='A',
                        help='saliency detector: u2net/basnet')
    parser.add_argument('--encoder-arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
    parser.add_argument('--decoder-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    # Dimension of positional encoding / condition for each decoder
    parser.add_argument('--cond_vec_len', default=128, type=int, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
    # Numbers of pooling layers used <--> How many multiscale-feature maps we have
    parser.add_argument('-pl', '--pool-layers', default=POOLING_LAYERS, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    # Numbers of coupling blocks <--> How many chained copupling blocks we want to have for each of the decoder
    parser.add_argument('-cb', '--coupling-blocks', default=COUPLING_BLOCKS, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')

    # Load weights of old model
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    # Parallel computing Args
    parser.add_argument('--workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    arg_parse_type = Config()
    args = parser.parse_args(namespace=arg_parse_type)
    return args
