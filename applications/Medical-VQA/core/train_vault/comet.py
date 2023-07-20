# Project:
#   VQA
# Description:
#   Comet ML related functions and classes
# Author: 
#   Sergio Tascon-Morales

from comet_ml import Experiment, ExistingExperiment
from misc.git import get_commit_hash

def get_new_experiment(config, path_config):
    if not config['comet_ml']:
        return None
    config_file_name = path_config.split("/")[-1].split(".")[0]
    comet_exp = Experiment() # Requires to have api key in config file in home folder  as suggested in the documentation
    comet_exp.add_tags([config['model'], config_file_name])
    comet_exp.log_parameters(config) 
    comet_exp.log_asset(path_config, file_name=path_config.split("/")[-1].split(".")[0]) # log yaml file to comet ml
    comet_exp.log_other('commit_hash', get_commit_hash()) # log commit hash
    return comet_exp

def get_existing_experiment(config):
    if not config['comet_ml']:
        return None 
    else:
        return ExistingExperiment(previous_experiment=config['experiment_key'])

def log_metrics(exp, metrics, epoch, to_log='all'):
    if exp is None:
        return
    if to_log == 'all':
        exp.log_metrics(metrics, epoch=epoch)
    else: # if only some of the metrics in the dictionary should be logged
        to_be_logged = {v: metrics[k] for k, v in to_log.items()}
        exp.log_metrics(to_be_logged, epoch=epoch)