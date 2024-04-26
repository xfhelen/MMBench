import torch
import sys
import os 
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import random
import argparse
import yaml
import torch.nn as nn
import torchvision
from models.unimodals.common_models import LeNet, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                    # print("shape", outs[i].shape)
            else:
                for i in range(len(inputs)):
                    outs.append(self.encoders[i](inputs[i]))
                    # print("shape", outs[i].shape)


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
    
            # print("out", out.shape)
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


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', default="normal", type=str, help='mode')
    parser.add_argument('--model_config', default='applications/customization/config.yaml', type=str, help='path to a yaml options file')
    args = parser.parse_args()
    return args

args = args_parser()

transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])    

def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def main():
    with open(args.path_config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    encoders = []
    if not config["have_time"]:
        if config['have_img'] :
            
            img = torch.zeros([config["img_size_x"], config["img_size_y"],config["channel"]]).to(device)
            if config["img_encoder"] == "Lenet":
                Lenet_channels = 6
                encoders.append(LeNet(config["channel"], Lenet_channels , 3).to(device))

            elif config["img_encoder"] == "Resnet":
                img = transforms(img)
                resnet = pretrained_resnet152().to(device)
                class Identity(torch.nn.Module):
                    def forward(self, input_: torch.Tensor) -> torch.Tensor:
                        return input_

                resnet.fc = Identity()  # Trick to avoid computing the fc1000 layer, as we don't need it here.

                avg_pool_value = resnet(img)

                video_feature = torch.mean(avg_pool_value,dim=0).view(-1,2048)
                encoders.append(video_feature)
        if config['have_text']:  
            text =           

    

