# Project:
#   VQA
# Description:
#   Train utils
# Author: 
#   Sergio Tascon-Morales

from os.path import join as jp
import numpy as np
import shutil
import torch
import os
from misc import dirs
from . import comet, logbook


def save_results(results, epoch_index, config, path_logs):
    # save tensor with indexes and answers produced by the model
    file_name = 'answers_epoch_' + str(epoch_index) + '.pt'
    path_answers = jp(path_logs, 'answers')
    dirs.create_folder(path_answers)
    torch.save(results, jp(path_answers, file_name))
    return 


def sync_if_parallel(config):
    if config['data_parallel']:
        torch.cuda.synchronize()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config, path_logs, lower_is_better=True):
        self.patience = config['patience']
        self.path_log = path_logs
        self.verbose = True
        self.counter = 0
        self.lower_is_better = lower_is_better 
        self.model_name = config['model']
        self.best_score_new = None
        self.early_stop = False
        self.best_score_old = np.Inf
        self.data_parallel = config['data_parallel']

    def __call__(self, metrics, metric_name, model, optimizer, epoch):

        score = metrics[metric_name]

        if self.best_score_new is None:
            self.best_score_new = score
            self.save_checkpoint(score, model, optimizer, self.path_log, metric_name, epoch)
        elif not self.lower_is_better and score <= self.best_score_new:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(score, model, optimizer, self.path_log, metric_name, epoch, best=False)
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.lower_is_better and score >= self.best_score_new:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            self.save_checkpoint(score, model, optimizer, self.path_log, metric_name, epoch, best=False)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score_new = score
            self.save_checkpoint(score, model, optimizer, self.path_log, metric_name, epoch)
            self.counter = 0

    def update_attributes(self, new_attributes):
        for k,v in new_attributes.items():
            setattr(self, k, v)

    def save_checkpoint(self, score, model, optimizer, path_experiment, metric_name, epoch, best=True):
        '''Saves model when validation metric improves.'''
        if best:
            info_file_name = 'best_checkpoint_info.pt'
            model_file_name = 'best_checkpoint_model.pt'
            optimizer_file_name = 'best_checkpoint_optimizer.pt'
            early_stop_file_name = 'best_checkpoint_early_stop.pt'
            if self.verbose:
                print(f'Metric {metric_name} improved ({self.best_score_old:.4f} --> {self.best_score_new:.4f}).  Saving model ...')
        else: 
            info_file_name = 'last_checkpoint_info.pt'
            model_file_name = 'last_checkpoint_model.pt'
            optimizer_file_name = 'last_checkpoint_optimizer.pt'
            early_stop_file_name = 'last_checkpoint_early_stop.pt'

        # save info
        info = {'epoch': epoch, 'model': self.model_name, metric_name: score}
        torch.save(info, jp(path_experiment, info_file_name))

        # save model parameters
        if not self.data_parallel:
            torch.save(model.state_dict(), jp(path_experiment, model_file_name))
        else:
            torch.save(model.module.state_dict(), jp(path_experiment, model_file_name))

        # save optimizer
        torch.save(optimizer.state_dict(), jp(path_experiment, optimizer_file_name))

        # save parameters of early stop
        torch.save(vars(self), jp(path_experiment, early_stop_file_name))

        # if it's best, make copy of pt files
        if best:
            shutil.copyfile(jp(path_experiment, info_file_name), jp(path_experiment, 'last_checkpoint_info.pt'))
            shutil.copyfile(jp(path_experiment, model_file_name), jp(path_experiment, 'last_checkpoint_model.pt'))
            shutil.copyfile(jp(path_experiment, optimizer_file_name), jp(path_experiment, 'last_checkpoint_optimizer.pt'))
            shutil.copyfile(jp(path_experiment, early_stop_file_name), jp(path_experiment, 'last_checkpoint_early_stop.pt'))

        self.best_score_old = self.best_score_new

        return


def initialize_experiment(config, model, optimizer, path_config, lower_is_better=False, classif=False):
    path_logs = jp(config['logs_dir'], config['dataset'], path_config.split("/")[-1].split(".")[0])
    if classif:
        path_logs = path_logs + '_classif'
    dirs.create_folder(path_logs)
    
    if config['train_from'] == 'scratch': # if training from scratch is desired
        dirs.clean_folder(path_logs) # remove old files if any
        # create comet ml experiment
        comet_experiment = comet.get_new_experiment(config, path_config)
        # instantiate early_stop object
        early_stopping = EarlyStopping(config, path_logs, lower_is_better=lower_is_better) # lower_is_better=False means metric to monitor E.S. should increase
        start_epoch = 1
        book = logbook.Logbook()

    elif config['train_from'] == 'last' or config['train_from'] == 'best': # from last saved checkpoint

        if config['comet_ml'] and config['experiment_key'] is None:
            raise ValueError("Please enter experiment key for comet experiment")

        # load info and model+optimizer parameters
        info = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_info.pt'))
        if torch.cuda.is_available():
            model_params = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_model.pt'))
            optimizer_params = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_optimizer.pt'))
        else:
            model_params = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_model.pt'), map_location=torch.device('cpu'))
            optimizer_params = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_optimizer.pt'), map_location=torch.device('cpu'))

        if not config['data_parallel']:
            model.load_state_dict(model_params)
        else:
            model.module.load_state_dict(model_params)
        optimizer.load_state_dict(optimizer_params)
        start_epoch = info['epoch'] + 1

        # resume comet experiment
        comet_experiment = comet.get_existing_experiment(config)

        # initialize early stopping
        early_stopping = EarlyStopping(config, path_logs, lower_is_better=lower_is_better)
        early_stopping_params = torch.load(jp(path_logs, config['train_from'] + '_checkpoint_early_stop.pt'))
        early_stopping.update_attributes(early_stopping_params)

        # create logbook
        book = logbook.Logbook()
        book.load_logbook(path_logs)

    else:
        raise ValueError("Wrong value for train_from option in config file. Options are best, last or scratch")

    return start_epoch, comet_experiment, early_stopping, book, path_logs