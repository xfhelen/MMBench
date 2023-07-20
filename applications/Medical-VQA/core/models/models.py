# Project:
#   VQA
# Description:
#   Model classes definition
# Author: 
#   Sergio Tascon-Morales

import torch
import torch.nn as nn
from torchvision import transforms
from .components.attention import apply_attention
from .components import image, text, attention, fusion, classification


class VQA_Base(nn.Module):
    # base class for simple VQA model
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__()
        self.visual_feature_size = config['visual_feature_size']
        self.question_feature_size = config['question_feature_size']
        self.pre_visual = config['pre_extracted_visual_feat']
        self.use_attention = config['attention']
        self.number_of_glimpses = config['number_of_glimpses']
        self.visual_size_before_fusion = self.visual_feature_size # 2048 by default, changes if attention

        # Create modules for the model

        # if necesary, create module for offline visual feature extraction
        if not self.pre_visual:
            self.image = image.get_visual_feature_extractor(config)

        # create module for text feature extraction
        self.text = text.get_text_feature_extractor(config, vocab_words)

        # if necessary, create attention module
        if self.use_attention:
            self.visual_size_before_fusion = self.number_of_glimpses*self.visual_feature_size
            self.attention_mechanism = attention.get_attention_mechanism(config)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # create multimodal fusion module
        self.fuser, fused_size = fusion.get_fuser(config['fusion'], self.visual_size_before_fusion, self.question_feature_size)

        # create classifier
        self.classifer = classification.get_classfier(fused_size, config)


    def forward(self, v, q):
        # extract text features
        q = self.text(q)
        # if required, extract visual features from visual input 
        if not self.pre_visual:
            v = self.image(v) # [B, 2048, 14, 14]

        # l2 norm
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        
        # extract text features
        q = self.text(q)
        print('VQA_BASE')
        print('v')
        print(v)
        print('q')
        print(q)
        return v 
        # if required, apply attention
        if self.use_attention:
            v = self.attention_mechanism(v, q) # should apply attention too
        else:
            v = self.avgpool(v).squeeze_() # [B, 2048]

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x
        

class VQARS_1(VQA_Base):
    # First model for region-based VQA, with single mask. Input image is multiplied with the mask to produced a masked version which is sent to the model as normal
    def __init__(self, config, vocab_words, vocab_answers):
        # call mom
        super().__init__(config, vocab_words, vocab_answers)

    # override forward method to accept mask
    def forward(self, options , v, q, m):
        if options == 'encoder' :
            if self.pre_visual:
                raise ValueError("This model does not allow pre-extracted features")
            else:
                v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

            # extract text features
            q = self.text(q)

        if options == 'fusion' :
            v = torch.zeros(2, 2048, 14, 14).to('cuda')
            q = torch.zeros(2, 1024).to('cuda')
            if self.use_attention:
                v = self.attention_mechanism(v, q) # should apply attention too
            else:
                v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]
            fused = self.fuser(v, q)
            return fused
        
        if options == 'head':
            fused = torch.zeros(2, 5120).to('cuda')
            x = self.classifer(fused)
            return x

        if options == 'normal':
        # if required, extract visual features from visual input 
            if self.pre_visual:
                raise ValueError("This model does not allow pre-extracted features")
            else:
                v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

            # extract text features
            q = self.text(q)
            # if required, apply attention
            if self.use_attention:
                v = self.attention_mechanism(v, q) # should apply attention too
            else:
                v = self.avgpool(v).squeeze_(dim=-1).squeeze_(dim=-1) # [B, 2048]

            # apply multimodal fusion
            fused = self.fuser(v, q)
            # apply MLP
            x = self.classifer(fused)

            return x


class SQuINT(VQARS_1):
    # SQuINTed version of model 1. See Selvaraju et al. 2020 (CVPR). Re-implemented for comparison purposes, since their code is not open-source.
    def __init__(self, config, vocab_words, vocab_answers):
        super().__init__(config, vocab_words, vocab_answers)

    # override forward so that attention maps are returned too
    def forward(self, v, q, m):
        # if required, extract visual features from visual input 
        if self.pre_visual:
            raise ValueError("This model does not allow pre-extracted features")
        else:
            v = self.image(torch.mul(v, m)) # [B, 2048, 14, 14]   MASK IS INCLUDED HERE

        # extract text features
        q = self.text(q)

        # if required, apply attention
        if self.use_attention:
            v, maps = self.attention_mechanism(v, q, return_maps=True) # should apply attention too
        else:
            raise ValueError("Attention is necessary for SQuINT")

        # apply multimodal fusion
        fused = self.fuser(v, q)

        # apply MLP
        x = self.classifer(fused)

        return x, maps

