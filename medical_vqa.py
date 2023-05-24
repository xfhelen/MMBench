# Project:
#   VQA
# Description:
#   Main train file for VQA model
# Author: 
#   Sergio Tascon-Morales


# IMPORTANT: All configurations are made through the yaml config file which is located in config/<dataset>/<file>.yaml. The path to this file is
#           specified using CLI arguments, with --path_config <path_to_yaml_file> . If you don't use comet ml, set the parameter comet_ml to False


import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import misc.io as io
from core.datasets import loaders_factory
from core.models import model_factory
from core.train_vault import criterions, optimizers, train_utils, looper, comet

# read config name from CLI argument --path_config
args = io.get_config_file_name()

def main():
    # read config file
    config = io.read_config(args.path_config)
    print('config done')
    # define device as gpu (if available) or cpu
    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')

    # load data
    train_loader, vocab_words, vocab_answers, index_unk_answer = loaders_factory.get_vqa_loader('train', config, shuffle=True) 
    val_loader = loaders_factory.get_vqa_loader('val', config) 
    print('dataload done')
    # create model
    model = model_factory.get_vqa_model(config, vocab_words, vocab_answers)

    # load weights for the loss function. The weights are provided in the dataset. See the script compute_answer_weights.py to check how they were computed
    if 'weighted_loss' in config:
        if config['weighted_loss']:
            answer_weights = io.read_weights(config) # if use of weights is required, read them from folder where they were previously saved using compute_answer_weights scripts
        else:
            answer_weights = None # If false, just set variable to None
    else:
        answer_weights = None
    # create criterion
    criterion = criterions.get_criterion(config, device, ignore_index = index_unk_answer, weights=answer_weights)

    # # create optimizer
    optimizer = optimizers.get_optimizer(config, model)

    # # create LR scheduler
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    # # initialize experiment
    start_epoch, comet_experiment, early_stopping, logbook, path_logs = train_utils.initialize_experiment(config, model, optimizer, args.path_config, lower_is_better=True)

    # # log config info
    logbook.log_general_info('config', config)

    # decide which functions are used for training depending on number of possible answers (binary or not)
    train, validate = looper.get_looper_functions(config)
    print('looper down')
    # Consistency loss term
    consisterm = criterions.ConsistencyLossTerm(config, vocab_words=train_loader.dataset.map_index_word)
    start_epoch = 0
    # train loop
    for epoch in range(start_epoch, config['epochs']+1):

        # train for one epoch
        # train_epoch_metrics = train(train_loader, model, criterion, optimizer, device, epoch, config, logbook, comet_exp=comet_experiment, consistency_term=consisterm)

        # log training metrics to comet, if required
        # comet.log_metrics(comet_experiment, train_epoch_metrics, epoch)

        # validate for one epoch
        validate(val_loader, model, criterion, device, epoch, config, logbook, comet_exp=comet_experiment, consistency_term=consisterm)
        break
        # log val metrics to comet, if required
        # comet.log_metrics(comet_experiment, val_epoch_metrics, epoch)

        # if val metric has stagnated, reduce LR
        # scheduler.step(val_epoch_metrics[config['metric_to_monitor']])

        # save validation answers for current epoch
        # train_utils.save_results(val_results, epoch, config, path_logs)
        # logbook.save_logbook(path_logs)

        # early_stopping(val_epoch_metrics, config['metric_to_monitor'], model, optimizer, epoch)

        # if patience was reached, stop train loop
        # if early_stopping.early_stop: 
        #     print("Early stopping")
        #     break

if __name__ == '__main__':
    main()
    print('finfished')
