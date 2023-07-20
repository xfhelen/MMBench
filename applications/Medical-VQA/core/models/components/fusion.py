# Project:
#   VQA
# Description:
#   Definitions for multimodal fusion schemes
# Author: 
#   Sergio Tascon-Morales

from torch import nn
import torch

def get_fuser(fusion_method, size_dim_1_a, size_dim1_b):

    if 'cat' in fusion_method:
        fuser = ConcatenationFusion()
        fused_size = size_dim_1_a + size_dim1_b
    elif 'mul' in fusion_method:
        fuser = HadamardFusion()
        fused_size = size_dim_1_a
    elif 'sum' in fusion_method:
        fuser = AdditionFusion()
        fused_size = size_dim_1_a
    else:
        raise ValueError("Unsupported fusion method")

    return fuser, fused_size

class ConcatenationFusion(nn.Module):
    # concatenates two [B, L] tensors
    def __init__(self):
        super().__init__()
        self.fuser = torch.cat
    def forward(self, x_1, x_2):
        return self.fuser((x_1, x_2), dim=1)  # [B, 2L]

class HadamardFusion(nn.Module):
    # Element-wise multiplication
    def __init__(self):
        super().__init__()
        self.fuser = torch.mul
    def forward(self, x_1, x_2):
        return self.fuser(x_1, x_2) # [B, L]

class AdditionFusion(nn.Module):
    # Addition of two tensors
    def __init__(self):
        super().__init__()
        self.fuser = torch.add
    def forward(self, x_1, x_2):
        return self.fuser(x_1, x_2) # [B, L]