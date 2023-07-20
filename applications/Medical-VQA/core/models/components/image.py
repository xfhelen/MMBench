# Project:
#   VQA
# Description:
#   Image feature extraction related functions and classes
# Author: 
#   Sergio Tascon-Morales

from torch import nn
from torchvision import models

def get_visual_feature_extractor(config):
    if 'resnet' in config['visual_extractor']:
        model  = ResNetExtractor(config['imagenet_weights'])
    else: 
        raise ValueError("Unknown model for visual feature extraction")
    return model

class ResNetExtractor(nn.Module):
    def __init__(self, imagenet):
        super().__init__()
        self.pre_trained = imagenet
        self.net_base = models.resnet152(pretrained = self.pre_trained)
        modules = list(self.net_base.children())[:-2] # ignore avgpool layer and classifier
        self.extractor = nn.Sequential(*modules)
        # freeze weights
        for p in self.extractor.parameters():
            p.requires_grad = False 

    def forward(self, x):
        x = self.extractor(x) # [B, 2048, 14, 14] if input is [B, 3, 448, 448]
        return x