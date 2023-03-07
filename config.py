from typing import TypedDict
import argparse

INPUT_SIZE = 256 # 256x256
BATCH_SIZE = 12
class Config(TypedDict):
    # Dataset
    dataset: str
    # General Config
    batch_size: int
    lr: float
    dropout: float
    epoch: int
    # Saliency config
    feature_extractor: str # String like: wide_resnet_50_2 | densenet,...
    local_saving_path: str
    output_bucket_path: str # Should be: /gcs/...
    # Debug printing config
    verbose: bool # Should only print debugs if true

def get_args():
    parser = argparse.ArgumentParser(description='Saliency_Flow')
    # General Mode
    parser.add_argument("--action-type", default='training', type=str, metavar='T',
                        help='training/testing (default: training)')
    parser.add_argument('--verbose', default=True, type=bool, metavar='G',
                        help='printing debug message (default: true)')
    # Model meta data config
    parser.add_argument('--dataset', default='plant_village', type=str, metavar='D', help='dataset name: plant_village (default: plant_village)')
    parser.add_argument('--bucket_save_path', default='FIX_DEFAULT', type=str, metavar='D',
                        help='where to save model output to')
    parser.add_argument('-inp', '--input-size', default=INPUT_SIZE, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')

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
    parser.add_argument('-lr','--learning-rate', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    # Model Architecture
    parser.add_argument('-encoder', '--feature-extractor', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
    parser.add_argument('-decoder', '--decoder-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    # Dimension of positional encoding / condition for each decoder
    parser.add_argument('-cond_vec_len', '--condition-vector-length', default=128, type=int, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
    # Numbers of pooling layers used <--> How many multiscale-feature maps we have
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    # Numbers of coupling blocks <--> How many chained copupling blocks we want to have for each of the decoder
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
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

    args = parser.parse_args()
    return args
