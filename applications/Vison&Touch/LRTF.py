import sys
import random
import yaml
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from models.fusions.robotics.sensor_fusion import roboticsConcat
from models.utils.helper_modules import Sequential2
from datasets.robotics.get_data import get_data
from models.fusions.common_fusions import LowRankTensorFusion
from models.unimodals.common_models import MLP
from models.unimodals.robotics.encoders import (ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder)
from models.eval_scripts.complexity import all_in_one_train

sys.path.append('/home/zhuxiaozhi/MMBench')
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
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

    def encoder_part(self,inputs):
        head_out = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                head_out.append(self.encoders[i]([inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                head_out.append(self.encoders[i](inputs[i]))
        self.reps = head_out
        return head_out
    
    def fusion_part(self, encoder_out):
        if self.has_padding:
            if isinstance(encoder_out[0], torch.Tensor):
                fusion_out = self.fuse(encoder_out)
            else:
                fusion_out = self.fuse([i[0] for i in encoder_out])
        else:
            fusion_out = self.fuse(encoder_out)
        self.fuseout = fusion_out
        return fusion_out
    
    def header_part(self, fusion_out,encoder_out,inputs):
        if type(fusion_out) is tuple:
            fusion_out = fusion_out[0]
        if self.has_padding and not isinstance(encoder_out[0], torch.Tensor):
            return self.head([fusion_out, inputs[1][0]])
        return self.head(fusion_out)

    def forward(self, inputs):
        """Apply MMDL to Layer Input.
        Args: inputs (torch.Tensor): Layer Input
        Returns: torch.Tensor: Layer Output
        """
        encoder_out = self.encoder_part(inputs)
        fusion_out = self.fusion_part(encoder_out)
        header_out = self.header_part(fusion_out,encoder_out,inputs)
        return header_out
        
def train(encoders, fusion, head, valid_dataloader, total_epochs, is_packed=False,input_to_float=True, track_complexity=True):

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    def _trainprocess():
        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for _ in range(total_epochs):
            model.eval()
            with torch.no_grad():
                for j in valid_dataloader:
                    if is_packed:
                        _ = model([[_processinput(i).to(device) for i in j[0]], j[1]])
                    else:
                        _ = model([_processinput(i).to(device) for i in j[:-1]])

    if track_complexity:
        all_in_one_train(_trainprocess, [model])
    else:
        _trainprocess()


def set_seeds(seed, use_cuda):
    """Set Seeds
    Args: seed (int): Sets the seed for numpy, torch and random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

class selfsupervised:
    def __init__(self, configs):
        use_cuda = True
        self.configs = configs
        self.device = torch.device("cuda" if use_cuda else "cpu")

        set_seeds(configs["seed"], use_cuda)

        self.encoders = [
            ImageEncoder(configs['zdim'], alpha=configs['vision']),
            ForceEncoder(configs['zdim'], alpha=configs['force']),
            ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
            DepthEncoder(configs['zdim'], alpha=configs['depth']),
            ActionEncoder(configs['action_dim']),
        ]
        self.fusion = Sequential2(roboticsConcat("noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))
        self.head = MLP(200, 128, 2)
        self.optimtype = optim.Adam
        self.loss_contact_next = nn.BCEWithLogitsLoss()

        self.val_loader = get_data(self.device, self.configs)

    def train(self):
        train(self.encoders, self.fusion, self.head, self.val_loader,total_epochs=1)

def main():
    with open('applications/Vison&Touch/training_default.yaml') as f:
        configs = yaml.load(f)

    selfsupervised(configs).train()

main()