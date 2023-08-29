import torch
import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import random
from PIL import Image
import argparse
from torch import nn
import pickle
softmax = nn.Softmax()
import torchvision.models as models
from models.training_structures.Supervised_Learning import test
from models.unimodals.common_models import MaxOut_MLP
from models.unimodals.common_models import Linear as mod_linear
from models.fusions.common_fusions import Concat
from torch.nn import Linear
import json
import sklearn.metrics
import numpy
import re
import sys
from datasets.avmnist.get_data import get_dataloader
from models.fusions.common_fusions import Concat
from models.unimodals.common_models import LeNet, MLP
# from robustness.text_robust import add_text_noise
# from robustness.visual_robust import add_visual_noise
from typing import *
import logging
import math
# from models.eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
# from models.eval_scripts.complexity import all_in_one_train, all_in_one_test
# from models.eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
import copy
from transformers import AlbertTokenizer, AlbertModel

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default="normal", type=str, help='mode')
    parser.add_argument('--model_name', type=str, default='avmnist_simple_late_fusion', help='name of model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    args = parser.parse_args()
    return args


args = args_parser()
####################################
#
# set configuration according params
#
####################################

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

traindata, validdata, testdata = get_dataloader(
        '/home/xucheng/xh/data/Multimedia/avmnist')
class MMDL(nn.Module):
    """Implements MMDL classifier."""
    
    def __init__(self, encoders, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs,options):
        if options == "normal":
            outs = []
            if self.has_padding:
                for i in range(len(inputs[0])):
                    outs.append(self.encoders[i](
                        [inputs[0][i], inputs[1][i]]))
                    print("shape", outs[i].shape)
            else:
                for i in range(len(inputs)):
                    outs.append(self.encoders[i](inputs[i]))
                    print("shape", outs[i].shape)


            self.reps = outs
            if self.has_padding:
                if isinstance(outs[0], torch.Tensor):
                    out = self.fuse(outs)
                else:
                    out = self.fuse([i[0] for i in outs])
            else:
                out = self.fuse(outs)

            self.fuseout = out
            if type(out) is tuple:
                out = out[0]
    
            print("out", out.shape)
            if self.has_padding and not isinstance(outs[0], torch.Tensor):
                return self.head([out, inputs[1][0]])
            return self.head(out)
            
        elif options == "encoder" :
            outs = []
            if self.has_padding:
                for i in range(len(inputs[0])):
                    outs.append(self.encoders[i](
                        [inputs[0][i], inputs[1][i]]))
            else:
                for i in range(len(inputs)):
                    outs.append(self.encoders[i](inputs[i]))
            return outs

        elif options == "fusion" :
            outs = []
            outs.append(torch.zeros([40,48]).to(device))
            outs.append(torch.zeros([40,192]).to(device))
            if self.has_padding:
                
                if isinstance(outs[0], torch.Tensor):
                    out = self.fuse(outs)
                else:
                    out = self.fuse([i[0] for i in outs])
            else:
                out = self.fuse(outs)
                return out
        
        elif options == "head" :
            outs = []
            outs.append(torch.zeros([40,48]).to(device))
            outs.append(torch.zeros([40,192]).to(device))
            out = torch.zeros([40,96].to(device))
            if self.has_padding and not isinstance(outs[0], torch.Tensor):
                return self.head([out, inputs[1][0]])
            return self.head(out)

options = args.options
if args.model_name=="avmnist_simple_late_fusion":
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*40, 100, 10).cuda()
    fusion = Concat().cuda()
elif args.model_name=="avmnist_tensor_matrix":
    from models.fusions.common_fusions import MultiplicativeInteractions2Modal
    channels = 3
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*32, 100, 10).cuda()
    fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*32, 'matrix', True).cuda()
elif args.model_name=="avmnist_multi_interac_matrix":
    from models.fusions.common_fusions import MultiplicativeInteractions2Modal
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
    head = MLP(channels*40, 100, 10).cuda()
    fusion = MultiplicativeInteractions2Modal(
    [channels*8, channels*32], channels*40, 'matrix')
elif args.model_name=="avmnist_cca":
    from models.utils.helper_modules import Sequential2
    from models.unimodals.common_models import Linear
    channels = 6
    encoders = [LeNet(1, channels, 3).cuda(), Sequential2(
    LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).cuda()]
    head = Linear(96, 10, xavier_init=True).cuda()
    fusion = Concat().cuda()

model = MMDL(encoders, fusion, head, has_padding=False).to(device)



def _processinput(inp):
    return inp.float()
with torch.no_grad():
    for j in testdata:
        model.eval()
        out = model([_processinput(i).float().to(device)
                            for i in j[:-1]], options)
        break

options == "normal"
with torch.no_grad():
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            './log/avmnist_agg'),
        record_shapes=True,
        with_stack=True)
    with prof as p: 
        for j in testdata:
            model.eval()
            out = model([_processinput(i).to(device)
                                for i in j[:-1]],options)
            p.step()