# Project:
#   VQA
# Description:
#   Train, validation and test loops
# Author: 
#   Sergio Tascon-Morales

import torch
from torch import nn
from . import train_utils
from metrics import metrics
import numpy as np

def train(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    
    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    for i, sample in enumerate(train_loader):
        batch_size = sample['question'].size(0)

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        answers = sample['answers'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output, answer)

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.vqa_accuracy(output, answers)

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    metrics_dict = {'loss_train': loss_epoch/len(train_loader.dataset), 'acc_train': acc_epoch/len(train_loader.dataset)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples


def train_dme(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    
    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        mask_a = sample['maskA'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output = model(visual, question, mask_a)
        train_utils.sync_if_parallel(config) #* necessary?

        # compute loss
        loss = criterion(output, answer)

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc} #! averaging by number of mini-batches

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples

def compute_mse(maps, flag, crit, device):
    # Function to compute MSE between attention maps (SQuINT)
    if torch.count_nonzero(flag) == 0: # if all flags are 0 (all questions are ind), return 0, otherwise NAN is generated.
        return torch.tensor(0)

    # separate into even and odd indexes (main vs sub or ind vs ind)
    main_maps = torch.index_select(maps, 0, torch.tensor([i for i in range(0, len(maps), 2)]).to(device))
    sub_maps = torch.index_select(maps, 0, torch.tensor([i for i in range(1, len(maps), 2)]).to(device))
    flag_reduced = torch.index_select(flag, 0, torch.tensor([i for i in range(0, len(flag), 2)]).to(device))
    mse = crit(main_maps, sub_maps)
    return torch.mean(mse[torch.where(flag_reduced>0)])

def train_dme_mainsub_squint(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    
    # In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0
    mse_accum = 0.0

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device)
        visual = sample['visual'].to(device)
        answer = sample['answer'].to(device)
        mask_a = sample['maskA'].to(device)
        flag = sample['flag'].to(device)

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        output, att_maps = model(visual, question, mask_a)
        train_utils.sync_if_parallel(config) #* necessary?

        # build term for inconsistency reduction
        mse = compute_mse(att_maps, flag, criterion2, device) 
        mse_accum += mse.item()

        loss = criterion1(output, answer) + config['lambda']*mse

        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc, 'mse_train': mse_accum/len(train_loader)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples


def train_dme_mainsub_consistrain(train_loader, model, criteria, optimizer, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    """Attempt to reduce inconsistencies in a different way than that proposed in Selvaraju et al"""

    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    # set train mode
    model.train()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    # if from second epoch, define beta for whole epoch
    if consistency_term.adaptive and epoch > 1:
        consistency_term.update_loss_params()

    for i, sample in enumerate(train_loader):

        # move data to GPU
        question = sample['question'].to(device) # [B, 23]
        visual = sample['visual'].to(device) # [B, 3, 448, 448]
        answer = sample['answer'].to(device) # [B]
        if 'maskA' in sample:
            mask_a = sample['maskA'].to(device)
        flag = sample['flag'].to(device) # [B]

        # clear parameter gradients
        optimizer.zero_grad()

        # get output from model
        if 'maskA' in sample:
            output = model(visual, question, mask_a)
        else:
            output = model(visual, question)
        train_utils.sync_if_parallel(config) #* necessary?

        # build term for inconsistency reduction
        non_avg_ce = criterion2(output, answer)
        
        # depending on function of consistency term, proceed as required
        if consistency_term.adaptive:
            # update max CE for main questions so that beta can be updated properly for next epoch
            consistency_term.log_ces_sub_main(non_avg_ce, flag, device)

            if epoch == 1: # in first epoch, do not include consistency term because you don't have a good estimate for beta
                loss = criterion1(output, answer)
            else:
                if config['consistency_function'] == 'fcn10':
                    q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device) # mq, ma, sq, sa
                else: 
                    q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
                loss = criterion1(output, answer) + q2_incons
                q2 += q2_incons.item()
        else:
            if config['consistency_function'] == 'fcn10':
                q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device) # mq, ma, sq, sa
            else: 
                q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
            loss = criterion1(output, answer) + q2_incons
            q2 += q2_incons.item()
            
        loss.backward()
        train_utils.sync_if_parallel(config) #* necessary?
        optimizer.step()
        train_utils.sync_if_parallel(config) #* necessary?

        running_loss += loss.item()
        if comet_exp is not None and i%10 == 9: # log every 10 iterations
            comet_exp.log_metric('loss_train_step', running_loss/10, step=len(train_loader)*(epoch-1) + i+1)
            running_loss = 0.0

        # compute accuracy
        acc = metrics.batch_strict_accuracy(output, answer)

        # laters: save to logger and print 
        loss_epoch += loss.item()
        acc_epoch += acc.item()

    if config['mainsub']:
        denominator_acc = 2*len(train_loader.dataset)
    else:
        denominator_acc = len(train_loader.dataset)

    metrics_dict = {'loss_train': loss_epoch/len(train_loader), 'acc_train': acc_epoch/denominator_acc, 'q2_train': q2/len(train_loader)}

    logbook.log_metrics('train', metrics_dict, epoch)

    return metrics_dict# returning average for all samples




def validate(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    # tensor to save results
    print('validate being run')
    results = torch.zeros((len(val_loader.dataset), 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    running_loss = 0.0
    acc_epoch = 0.0

    offset = 0
    options = 'encoder'
    if options == 'encoder' :
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                batch_size = sample['question'].size(0)
                # move data to GPU
                question = sample['question'].to(device)
                visual = sample['visual'].to(device)
                answer = sample['answer'].to(device)
                answers = sample['answers'].to(device)
                question_indexes = sample['question_id'] # keep in cpu

                # get output
                output = model(visual, question)
                # compute loss
                # loss = criterion(output, answer)

                # # compute accuracy
                # acc = metrics.vqa_accuracy(output, answers)

                # # save answer indexes and answers
                # sm = nn.Softmax(dim=1)
                # probs = sm(output)
                # _, pred = probs.max(dim=1) 
                # results[offset:offset+batch_size,0] = question_indexes 
                # results[offset:offset+batch_size,1] = pred
                # offset += batch_size

                # loss_epoch += loss.item()
                # acc_epoch += acc.item()
    else :
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                batch_size = sample['question'].size(0)
                # move data to GPU
                question = sample['question'].to(device)
                visual = sample['visual'].to(device)
                answer = sample['answer'].to(device)
                answers = sample['answers'].to(device)
                question_indexes = sample['question_id'] # keep in cpu

                # get output
                output = model(visual, question)


    metrics_dict = {'loss_val': loss_epoch/len(val_loader.dataset), 'acc_val': acc_epoch/len(val_loader.dataset)}
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples

def validate_dme(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=None, consistency_term=None):

    #if config['mainsub']:
    #    denominator_acc = 2*len(val_loader.dataset)
    #else:
    denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask_a = sample['maskA'].to(device)

            # get output
            output = model(visual, question, mask_a)

            if 'squint' in config: # special case for squint
                if config['squint']:
                    output = output[0]
                

            # compute loss
            if isinstance(criterion, tuple):
                loss = criterion[0](output, answer)
            else:
                loss = criterion(output, answer)

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc} #! averaging by number of mini-batches
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples


def validate_dme_mainsub_squint(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
 #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    mse_accum = 0.0

    offset = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            batch_size = sample['question'].size(0)

            # move data to GPU
            question = sample['question'].to(device)
            visual = sample['visual'].to(device)
            answer = sample['answer'].to(device)
            question_indexes = sample['question_id'] # keep in cpu
            mask_a = sample['maskA'].to(device)
            flag = sample['flag'].to(device)

            # get output
            output, att_maps = model(visual, question, mask_a)
            train_utils.sync_if_parallel(config) #* necessary?

            # build term for inconsistency reduction
            mse = compute_mse(att_maps, flag, criterion2, device)
            mse_accum += mse.item()

            loss = criterion1(output, answer) + config['lambda']*mse

            # compute accuracy
            acc = metrics.batch_strict_accuracy(output, answer)

            # save answer indexes and answers
            sm = nn.Softmax(dim=1)
            probs = sm(output)
            _, pred = probs.max(dim=1)
        
            results[offset:offset+batch_size,0] = question_indexes
            results[offset:offset+batch_size,1] = pred
            offset += batch_size

            loss_epoch += loss.item()
            acc_epoch += acc.item()

    metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc, 'mse_val': mse_accum/len(val_loader)} 
    if logbook is not None:
        logbook.log_metrics('val', metrics_dict, epoch)

    return metrics_dict, results # returning averages for all samples



def validate_dme_mainsub_consistrain(val_loader, model, criteria, device, epoch, config, logbook, comet_exp=None, consistency_term=None):
    print('being visited')
    #* In this case criteria contains two cross entropies: one as usual for right answers and a second one without reduction
    criterion1, criterion2 = criteria

    if config['mainsub']:
        denominator_acc = 2*len(val_loader.dataset)
    else:
        denominator_acc = len(val_loader.dataset)

    # tensor to save results
    results = torch.zeros((denominator_acc, 2), dtype=torch.int64)

    # set evaluation mode
    model.eval()

    # Initialize variables for collecting metrics from all batches
    loss_epoch = 0.0
    acc_epoch = 0.0
    q2 = 0.0

    offset = 0
    with torch.no_grad():
        options = 'encoder'
        if options == 'encoder' or options == 'normal':
            for i, sample in enumerate(val_loader):
                batch_size = sample['question'].size(0)

                # move data to GPU
                question = sample['question'].to(device)
                visual = sample['visual'].to(device)
                answer = sample['answer'].to(device)
                question_indexes = sample['question_id'] # keep in cpu
                if 'maskA' in sample:
                    mask_a = sample['maskA'].to(device)
                flag = sample['flag'].to(device)

                # get output
                if 'maskA' in sample:
                    output = model(visual, question, mask_a)
                else:
                    output = model(visual, question)
                if i == 10:
                    break
        else : 
            for i in range(50):
                output = model(1,2,3)
                if i == 50:
                    break
    #         # build term for inconsistency reduction
    #         non_avg_ce = criterion2(output, answer)

    #         if consistency_term.adaptive:
    #             if epoch == 1: # in first epoch, do not include consistency term because you don't have a good estimate for beta
    #                 loss = criterion1(output, answer)
    #             else:
    #                 if config['consistency_function'] == 'fcn10':
    #                     q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device)
    #                 else:
    #                     q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
    #                 loss = criterion1(output, answer) + q2_incons
    #                 q2 += q2_incons.item()
    #         else:
    #             if config['consistency_function'] == 'fcn10':
    #                 q2_incons = config['lambda']*consistency_term.compute_loss_term_nlp(question, output, flag, device)
    #             else:
    #                 q2_incons = config['lambda']*consistency_term.compute_loss_term(non_avg_ce, flag, device)
    #             loss = criterion1(output, answer) + q2_incons
    #             q2 += q2_incons.item()

    #         # compute accuracy
    #         acc = metrics.batch_strict_accuracy(output, answer)

    #         # save answer indexes and answers
    #         sm = nn.Softmax(dim=1)
    #         probs = sm(output)
    #         _, pred = probs.max(dim=1)
        
    #         results[offset:offset+batch_size,0] = question_indexes
    #         results[offset:offset+batch_size,1] = pred
    #         offset += batch_size

    #         loss_epoch += loss.item()
    #         acc_epoch += acc.item()

    # metrics_dict = {'loss_val': loss_epoch/len(val_loader), 'acc_val': acc_epoch/denominator_acc, 'q2_val': q2/len(val_loader)} 
    # if logbook is not None:
    #     logbook.log_metrics('val', metrics_dict, epoch)
    return  # returning averages for all samples
    # return metrics_dict, results # returning averages for all samples


def get_looper_functions(config, classif=False, test=False):
    # function to define which functions are used to train and validate for one epoch, depending on the configurations
    if test:
        return validate_dme, validate_dme

    if config['mainsub']:
        # mainsub = True means batches will be produced with pairs of main-sub questions
        if 'squint' in config:
            if config['squint']:
                train_fn = train_dme_mainsub_squint
                val_fn = validate_dme_mainsub_squint
            else:
                train_fn = train_dme_mainsub_consistrain
                val_fn = validate_dme_mainsub_consistrain                   
        else:
            train_fn = train_dme_mainsub_consistrain
            val_fn = validate_dme_mainsub_consistrain
    elif config['dataset'] == 'vqa2':
        # if dataset is VQA2 (normal training, not for our method)
        train_fn = train 
        val_fn = validate
    else:
        train_fn = train_dme 
        val_fn = validate_dme

    return train_fn, val_fn
