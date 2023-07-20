# Project:
#   VQA
# Description:
#   Optimizer definition
# Author: 
#   Sergio Tascon-Morales

import torch

def get_optimizer(config, model):

    if 'adam' in config['optimizer']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'adadelta' in config['optimizer']:
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'rmsprop' in config['optimizer']:
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])
    elif 'sgd' in config['optimizer']:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), config['learning_rate'])

    return optimizer