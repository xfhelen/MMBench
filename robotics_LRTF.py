import sys
import os
sys.path.insert(0, os.getcwd())
from models.fusions.robotics.sensor_fusion import SensorFusionSelfSupervised, roboticsConcat
from models.utils.helper_modules import Sequential2
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.robotics.get_data import get_data
from models.fusions.common_fusions import LowRankTensorFusion
from models.training_structures.Supervised_Learning import train, test
from models.unimodals.robotics.decoders import ContactDecoder
from models.unimodals.common_models import MLP
from models.unimodals.robotics.encoders import (
    ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder,
)
from models.eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from models.eval_scripts.complexity import all_in_one_train, all_in_one_test
from tqdm import tqdm
import yaml
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time
from torch import profiler
import math
import random
import copy
import time
def set_seeds(seed, use_cuda):
    """Set Seeds

    Args:
        seed (int): Sets the seed for numpy, torch and random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


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

    def forward(self, inputs):
        """Apply MMDL to Layer Input.
        Args:inputs (torch.Tensor): Layer Input
        Returns:torch.Tensor: Layer Output
        """
        with profiler.record_function("LINEAR PASS"):
            outs = []
        for i in range(len(inputs)):
        self.reps = outs
        # with profiler.record_function("FUSION"):
        out = self.fuse(outs)
#         return out
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        # with profiler.record_function("HEAD"):
        # print(out.shape)
#         out = torch.zeros(64,200).to(device)
        return self.head(out)

use_cuda = True
with open('training_default.yaml') as f:
    configs = yaml.load(f,Loader=yaml.FullLoader)
set_seeds(configs["seed"], use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
# Parse args
# train_dataloader, valid_dataloader = get_data(
    # device, configs, "./datasets/robotics/triangle_real_data")
valid_dataloader = get_data(
    device, configs, "./datasets/robotics/triangle_real_data")

encoders = [
    ImageEncoder(configs['zdim'], alpha=configs['vision']),
    ForceEncoder(configs['zdim'], alpha=configs['force']),
    ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
    DepthEncoder(configs['zdim'], alpha=configs['depth']),
    ActionEncoder(configs['action_dim']),
]
fusion = Sequential2(roboticsConcat(
        "noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))

head = MLP(200, 128, 2)
optimtype = optim.Adam
loss_state = nn.BCEWithLogitsLoss()
model = MMDL(encoders, fusion, head, has_padding=False).to(device)

total_epochs = 1
lr=configs['lr']
additional_optimizing_modules=[]
is_packed=False
early_stop=False
task="classification"
optimtype=torch.optim.RMSprop
lr=0.001
weight_decay=0.0
objective=nn.CrossEntropyLoss()
auprc=False
save='best.pt'
validtime=False
objective_args_dict=None
input_to_float=True
clip_val=8
track_complexity=True

def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size())-1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(objective) == nn.L1Loss:
        return objective(pred, truth.float().to(device))
    else:
        return objective(pred, truth, args)
def _trainprocess():
    additional_params = []
    for m in additional_optimizing_modules:
        additional_params.extend(
            [p for p in m.parameters() if p.requires_grad])
    op = optimtype([p for p in model.parameters() if p.requires_grad] +
                    additional_params, lr=lr, weight_decay=weight_decay)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    def _processinput(inp):
        if input_to_float:
            return inp.float()
        else:
            return inp

    for epoch in range(total_epochs):
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []
            cat = 0
            for j in valid_dataloader:
                if cat == 10 :
                    break
                cat = cat + 1
                out = model([_processinput(i).to(device)
                                for i in j[:-1]])
            print('down')
            
_trainprocess()

def _processinput(inp):
    if input_to_float:
        return inp.float()
    else:
        return inp

