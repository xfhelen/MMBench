# Project:
#   VQA
# Description:
#   Loss function definitions and getter
# Author: 
#   Sergio Tascon-Morales

from torch import nn
import numpy as np
import torch
from ..models import model_factory
import os

def get_criterion(config, device, ignore_index = None, weights = None):
    # function to return a criterion. By default I set reduction to 'sum' so that batch averages are not performed because I want the average across the whole dataset

    if 'crossentropy' in config['loss']:
        if weights is not None:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean', weight=weights).to(device)        
        else:
            crit = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum').to(device)
    elif 'bce' in config['loss']:
        if weights is not None:
            crit = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        else:
            crit = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    else:
        raise ValueError("Unknown loss function.")
    
    if 'mainsub' in config:
        if config['mainsub']:
            # if mainsub is set to true (meaning batches are generated in pairs of main-sub questions) then create a second criterion that will measure the cross-entropy between pairs
            if 'squint' in config: # if squint is True, second loss term is MSE
                if config['squint']:
                    mse = nn.MSELoss(reduction='none').to(device)
                    return crit, mse
                else:
                    # create second criterion
                    if config['num_answers'] == 2:
                        ce = nn.BCEWithLogitsLoss(reduction='none').to(device)
                    else:
                        ce = nn.CrossEntropyLoss(reduction='none').to(device)
                    return crit, ce           
            else:
                # create second criterion
                if config['num_answers'] == 2:
                    ce = nn.BCEWithLogitsLoss(reduction='none').to(device)
                else:
                    ce = nn.CrossEntropyLoss(reduction='none').to(device)
                return crit, ce
        else:
            return crit
    else:
        return crit


def Q2_score(scores_main, gt_main, scores_sub, gt_sub, suit):
    # create softmax
    sm = nn.Softmax(dim=1)
    probs_main = sm(scores_main)
    _, ans_main = probs_main.max(dim=1) 
    probs_sub = sm(scores_sub)
    _, ans_sub = probs_sub.max(dim=1)

    # Q2 score is implemented to measure the number of Q2 inconsistencies within the batch, meaning in how many cases the model predicted the 
    # main question correctly but failed to answer the associated sub-question correctly. 
    q2_score = torch.sum(torch.logical_and(torch.eq(ans_main, gt_main), torch.logical_not(torch.eq(ans_sub, gt_sub)))*suit)/gt_main.shape[0]

    return q2_score



def fcn2(non_avg_ce, flag, device, gamma=0.5, exp=True):
    # Function corresponding to our proposed loss term. 
    # See paper for more information.

    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(0, len(non_avg_ce), 2)]).to(device))
    sub_ce = torch.index_select(non_avg_ce, 0, torch.tensor([i for i in range(1, len(non_avg_ce), 2)]).to(device))
    # summarize flag vector to be taken in same indexes as main_ce
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    if exp:
        func = torch.exp(sub_ce*(gamma - main_ce))-1
    else:
        func = sub_ce*(gamma - main_ce)
    #torch.exp(alpha*sub_ce)*torch.cos(beta*main_ce) - 1 # subtract 1 so that loss term is 0 at (0,0)
    relu = nn.ReLU()
    relued = relu(func)
    return torch.mean(relued[torch.where(flag_reduced>0)])


def fcn4():
    # dummy function to test baseline
    return


class ConsistencyLossTerm(object):
    """Class for consistency loss term with different functions

    """

    def __init__(self, config, vocab_words=None):
        if 'consistency_function' not in config:
            return
        else:
            self.fcn = config['consistency_function']
            
            if self.fcn not in globals():
                raise ValueError("Unknown function")
            else:
                self.loss_term_fcn = globals()[self.fcn]

            if 'adaptive' in config:
                self.adaptive = config['adaptive']
                self.min_ce_previous_epoch = np.inf # define first min ce for main-questions as 0 (will be modified after first epoch)
            else:
                self.adaptive = False

            if self.fcn == 'fcn2' or self.fcn == 'fcn8' or self.fcn == 'fcn9':
                self.gamma = config['gamma']
                self.exp = config['exp']
            elif self.fcn == 'fcn4':
                pass            
            else:
                raise ValueError

    def compute_loss_term(self, non_avg_ce, flag, device):
        # Depending on function, compute loss term accordingly
        if self.fcn == 'fcn2' or self.fcn == 'fcn8' or self.fcn == 'fcn9':
            return self.loss_term_fcn(non_avg_ce, flag, device, gamma=self.gamma, exp=self.exp)
        elif self.fcn == 'fcn4':
            return torch.tensor(0) # return 0 (no additional loss term)
        else:
            raise ValueError('Unknown function. If youre using the NLP model, use compute_loss_term_nlp instead of compute_loss_term')


