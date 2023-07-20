# Project:
#   VQA
# Description:
#   Logbook class to track results
# Author: 
#   Sergio Tascon-Morales

import os
import json
from os.path import join as jp

class Logbook(object):
    def __init__(self, init_info = None):
        if init_info is not None:
            self.book = init_info
        else:
            self.book = {'general': {}, 'train': {}, 'val': {}} # by default it will have some general info, some training info and some val info
            
    def log_metric(self, stage, metric_name, value, epoch):
        if stage != 'train' and stage != 'val':
            raise ValueError("stage must be either train or val")
        # check if metric was already created previously. If not, create it and log
        if metric_name not in self.book[stage]:
            self.book[stage][metric_name] = {}
            self.book[stage][metric_name][epoch] = value
        else: # if metric already exists, check if it was already reported for the epoch index, then report
            # check if metric was already logged for the given epoch (just as sanity check)
            if epoch in self.book[stage][metric_name]:
                print("Warning: Entry already exists for given epoch index " + str(epoch))
            self.book[stage][metric_name][epoch] = value

    def log_metrics(self, stage, metrics, epoch):
        # receives dictionary
        for k,v in metrics.items():
            self.log_metric(stage, k, v, epoch)

    def log_general_info(self, key, value):
        # log anything as general info to the book
        self.book['general'][key] = value

    def save_logbook(self, path):
        # path is just the path to the folder where the json file should be stored, not the full path to the file
        with open(jp(path, 'logbook.json'), 'w') as f:
            json.dump(self.book, f)

    def load_logbook(self, path):
        if not os.path.exists(jp(path, 'logbook.json')):
            raise Exception("File logbook.json does not exists at " + path)
        else:
            with open(jp(path, 'logbook.json')) as f:
                self.book = json.load(f)

