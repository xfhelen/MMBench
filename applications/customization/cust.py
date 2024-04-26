import torch
import sys
import os 
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import random
import argparse
import yaml


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default="normal", type=str, help='mode')
    parser.add_argument('--model_name', type=str, default='avmnist_simple_late_fusion', help='name of model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    args = parser.parse_args()
    return args
parser = argparse.ArgumentParser(
    description='Read config file name',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)