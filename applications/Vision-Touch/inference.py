import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import random
import yaml
import torch.nn as nn
import torch
import numpy as np
import argparse
import onnx
from models.fusions.robotics.sensor_fusion import roboticsConcat
from models.utils.helper_modules import Sequential2
from datasets.robotics.get_data import get_data
from models.fusions.common_fusions import LowRankTensorFusion
from models.fusions.common_fusions import Concat
from models.unimodals.common_models import MLP
from models.unimodals.common_models import Identity
from models.unimodals.robotics.encoders import (ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder)
from models.eval_scripts.complexity import all_in_one_train


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--options', default="normal", type=str, help='choose the model part') 
args = parser.parse_args()
options = args.options


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
        print(head_out[-1].shape)
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

    def manual_encoder_out(self,inputs):
        outs = []
        for i in range(len(inputs)):
            if i == 0:
                outss = []
                outss.append(torch.ones(64,256,1).to(device))
                outsss = []
                outsss.append(torch.ones(64,16,64,64).to(device))
                outsss.append(torch.ones(64,32,32,32).to(device))
                outsss.append(torch.ones(64,64,16,16).to(device))
                outsss.append(torch.ones(64,64,8,8).to(device))
                outsss.append(torch.ones(64,128,4,4).to(device))
                outsss.append(torch.ones(64, 128, 2, 2).to(device))
                outss.append(outsss)
                outs.append(outss)
                continue
            if i == 3:
                outss = []
                outss.append(torch.ones(64,256,1).to(device))
                outss.append(torch.ones(64, 32, 64, 64).to(device))
                outss.append(torch.ones(64, 64, 32, 32).to(device))
                outss.append(torch.ones(64, 64, 16, 16).to(device))
                outss.append(torch.ones(64, 64, 8, 8).to(device))
                outss.append(torch.ones(64, 128, 4, 4).to(device))
                outss.append(torch.ones(64, 128, 2, 2).to(device))
                outs.append(outss)
                continue
            if i == 4:
                outs.append(torch.ones(64, 32).to(device))
                continue
            outs.append(torch.ones(64, 256, 1).to(device))
        return outs

    def forward(self, inputs):
        if options == 'normal':
            encoder_out = self.encoder_part(inputs) 
            fusion_out = self.fusion_part(encoder_out)
            return self.header_part(fusion_out,encoder_out,inputs)
        elif options == 'encoder':
            return self.encoder_part(inputs)
        elif options == 'fusion':
            encoder_out = self.manual_encoder_out(inputs)
            return self.fusion_part(encoder_out)
        elif options == 'head':
            fusion_out = torch.zeros(64,200).to(device)
            encoder_out = self.manual_encoder_out(inputs)
            return self.header_part(fusion_out,encoder_out,inputs)
        else:
            print('Please choose the right options from normal, encoder, fusion, head')
            exit()
        
def train(encoders, fusion, head, valid_dataloader, total_epochs, is_packed=False,input_to_float=True, track_complexity=True):

    model = MMDL(encoders, fusion, head, has_padding=is_packed).to(device)

    def _trainprocess():
        def _processinput(inp):
            if input_to_float:
                # print(inp)
                return inp.float()
            else:
                return inp

        for _ in range(total_epochs):
            model.eval()
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('applications/Vision-Touch/log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    for j in valid_dataloader:
                        
                        if is_packed:
                            _ = model([[_processinput(i).to(device) for i in j[0]], j[1]])
                        else:
                            _ = model([_processinput(i).to(device) for i in j[:-1]])
                        prof.step()

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

def main():
    with open('applications/Vision-Touch/training_default.yaml') as f:
        configs = yaml.load(f,yaml.FullLoader)
    use_cuda = True
    configs = configs
    device = torch.device("cuda" if use_cuda else "cpu")

    set_seeds(configs["seed"], use_cuda)

    encoders = [
        ImageEncoder(configs['zdim'], alpha=configs['vision']),
        ForceEncoder(configs['zdim'], alpha=configs['force']),
        ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
        DepthEncoder(configs['zdim'], alpha=configs['depth']),
        ActionEncoder(configs['action_dim']),
    ]
    fusion = Sequential2(roboticsConcat("noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))
    head = MLP(200, 128, 2)

    val_loader = get_data(device, configs)

    train(encoders, fusion, head, val_loader,total_epochs=1)

#Function to Convert to ONNX 
def Convert_ONNX(): 
    with open('applications/Vision-Touch/training_default.yaml') as f:
        configs = yaml.load(f,yaml.FullLoader)
    use_cuda = True
    configs = configs

    set_seeds(configs["seed"], use_cuda)

    # input data size
    # torch.Size([64, 128, 128, 3])
    # torch.Size([64, 6, 32])
    # torch.Size([64, 8])
    # torch.Size([64, 1, 128, 128])
    # torch.Size([64, 4])
    encoders = [
        ImageEncoder(configs['zdim'], alpha=configs['vision']),
        ForceEncoder(configs['zdim'], alpha=configs['force']),
        ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
        DepthEncoder(configs['zdim'], alpha=configs['depth']),
        ActionEncoder(configs['action_dim']),
    ]
    fusion = Sequential2(roboticsConcat("noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))
    fusion.eval()
    # fusion = Sequential2(roboticsConcat("noconcat"), Concat())
    head = MLP(200, 128, 2)
    # head = Identity()
    val_loader = get_data(device, configs)

    model = MMDL(encoders, fusion, head).to(device)
    model.eval() 

    data = next(iter(val_loader))
    true_input = [i.float().to(device) for i in data[:-1]]
    # for i in range(len(true_input)):
    #     print(true_input[i].shape)
    dummy_input = [torch.randn(true_input[i].shape) for i in range(len(true_input))]

    torch.onnx.export(model, true_input, "./applications/Vision-Touch/model.onnx", export_params=True, opset_version=12)


if __name__ == "__main__":
    main()
    # Convert_ONNX()