# Project:
#   VQA
# Description:
#   utils for vqa model
# Author: 
#   Sergio Tascon-Morales


def expand_like_2D(to_expand, reference):
    # expands input with dims [B, K] to dimensions of reference which are [B, K, M, M]
    expanded = to_expand.unsqueeze_(2).unsqueeze_(3).expand_as(reference)
    return expanded