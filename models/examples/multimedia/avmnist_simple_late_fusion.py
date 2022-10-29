import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
import argparse
traindata, validdata, testdata = get_dataloader(
    '/home/xucheng/xh/data/Multimedia/avmnist')



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_name', help='name of model')
    parser.add_argument('--model_index', type=int, default=-1, help='name of model')
    parser.add_argument('--gpu', type=int, default=2, help='name of model')
    parser.add_argument('--lr', type=float, default=8e-3, help='name of model')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='name of model')
    parser.add_argument('--random_seed', type=int, default=20, help='name of model')
    parser.add_argument('--early_exit_step', type=int, default=15, help='name of model')
    args = parser.parse_args()
    return args
args= args_parser()

filename="./models_save/best_{}_{}.pth".format(args.model_name,args.model_index)
if args.model_name=="avmnist_simple_late_fusion":
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*40, 100, 10).cuda()
    fusion = Concat().cuda()
elif args.model_name=="avmnist_tensor_matrix":
    from fusions.common_fusions import MultiplicativeInteractions2Modal
    channels = 3
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*32, 100, 10).cuda()
    fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*32, 'matrix', True).cuda()
elif args.model_name=="avmnist_multi_interac_matrix":
    from fusions.common_fusions import MultiplicativeInteractions2Modal
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*40, 100, 10).cuda()
    fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')
elif args.model_name=="avmnist_cca":
    from utils.helper_modules import Sequential2
    from unimodals.common_models import Linear
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), Sequential2(
    LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).cuda()]
    head = Linear(96, 10, xavier_init=True).cuda()
    fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 30,
    save=filename,optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)
"""
print("Testing:")
model = torch.load(filename).cuda()
test(model, testdata, no_robust=True)
"""