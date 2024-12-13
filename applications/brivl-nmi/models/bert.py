import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, BertConfig


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        config = BertConfig()
        self.bert = BertModel(config)
        # self.bert =  AutoModel.from_pretrained(args.ENCODER) 
        
    def forward(self, x):
        y = self.bert(x, return_dict=True).last_hidden_state
        return y
