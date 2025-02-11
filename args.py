import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from myfunctions import seed_torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--data_path', type=str,
                    default="./datasets/247data/",
                    help='The directory containing the EV charging data.')
parser.add_argument('--LOOK_BACK', type=int, default=12,
                    help='Number of time step of the Look Back Mechanism.')
parser.add_argument('--predict_time', type=int, default=3,
                    help='Number of time step of the predict time.')
parser.add_argument('--nodes', type=int, default=247,
                    help='Number of areas.')
parser.add_argument('--max_epochs', type=int, default=2000,
                    help='The max meta-training epochs.')
parser.add_argument('--patience', type=int, default=50,
                    help='The patience of early stopping.')
parser.add_argument('--training_rate', type=float, default=0.6,
                    help='The rate of training set.')
parser.add_argument('--valid_rate', type=float, default=0.1,
                    help='The rate of valid set.')
parser.add_argument('--method', type=str,
                    default="gumbelsoftmax2",
                    help='The using method.')



args = parser.parse_args(args=[]) # jupyter
# args = parser.parse_args()      # python

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(2024)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
